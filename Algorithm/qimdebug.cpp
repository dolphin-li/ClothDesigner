#include "qimdebug.h"
#include <assert.h>

#include <QWidget>
#include <qmap.h>
#include <QPainter>
#include <QMouseEvent>
#include <QKeyEvent>
#include <qfiledialog.h>
#include <iostream>

#define define_is_type(v)\
template<typename T>\
struct _is_##v\
{\
	static const bool value = false;\
};\
template<>\
struct _is_##v<v>\
{\
	static const bool value = true;\
};
define_is_type(uchar)
define_is_type(int)
define_is_type(float)

#define is_type(T, v) _is_##v<T>::value


enum QImageDebugger_DataType{
	QImageDebugger_Byte,
	QImageDebugger_Int,
	QImageDebugger_Float,
};

class QImdebugger : public QWidget
{
public:
	QImdebugger(QWidget *parent) : QWidget(parent)
	{
		m_img = nullptr;
		m_channels = 0;
		m_reverse_y = false;
		setMouseTracking(true);
		setWindowTitle("QImageDebug");
	}
	~QImdebugger()
	{
		if (m_img)
			delete m_img;
	}

	void setName(QString name)
	{
		m_name = name;
		setWindowTitle(m_name);
	}

	void resizeImage(int w, int h)
	{
		if (m_img)
		{
			if (!(m_img->width() == w && m_img->height() == h))
			{
				delete m_img;
				m_img = new QImage(w, h, QImage::Format::Format_ARGB32);
			}
		}
		else
			m_img = new QImage(w, h, QImage::Format::Format_ARGB32);
	}

	template<class T>
	bool setImage(const T* data, int w, int h, QImdebugType type, bool reverse_y)
	{
		resizeImage(w, h);
		m_viewTransform.reset();
		m_reverse_y = reverse_y;

		if (is_type(T, float))
			m_dataType = QImageDebugger_Float;
		else if (is_type(T, int))
			m_dataType = QImageDebugger_Int;
		else if (is_type(T, uchar))
			m_dataType = QImageDebugger_Byte;
		else
		{
			printf("QImageDebugger: non-supported type!\n");
			return false;
		}
		m_type = type;

		T minV = std::numeric_limits<T>::max();
		T maxV = std::numeric_limits<T>::lowest();

		switch (type)
		{
		case QImdebugType_RGBA:
			m_channels = 4;
			for (int y = 0; y < h; y++)
			{
				const T* yptr = data + (reverse_y?h-1-y:y) * w * 4;
				for (int x = 0; x < w; x++)
					m_img->setPixel(x, y, qRgba(yptr[x * 4], yptr[x * 4 + 1], yptr[x * 4 + 2], yptr[x * 4 + 3]));
			}
			break;
		case QImdebugType_RGB:
			m_channels = 3;
			for (int y = 0; y < h; y++)
			{
				const T* yptr = data + (reverse_y ? h - 1 - y : y) * w * 3;
				for (int x = 0; x < w; x++)
					m_img->setPixel(x, y, qRgb(yptr[x * 3], yptr[x * 3 + 1], yptr[x * 3 + 2]));
			}
			break;
		case QImdebugType_GRAY:
			m_channels = 1;
			for (int y = 0; y < h; y++)
			{
				const T* yptr = data + (reverse_y ? h - 1 - y : y) * w;
				for (int x = 0; x < w; x++)
					m_img->setPixel(x, y, qRgb(yptr[x], yptr[x], yptr[x]));
			}
			break;
		case QImdebugType_INDEX:
			m_channels = 1;
			for (int i = w*h - 1; i >= 0; i--)
			{
				maxV = std::max(maxV, data[i]);
				minV = std::min(minV, data[i]);
			}
			for (int y = 0; y < h; y++)
			{
				const T* yptr = data + (reverse_y ? h - 1 - y : y) * w;
				for (int x = 0; x < w; x++)
				{
					T val; 
					if (maxV == minV)
						val = 128;
					else
						val = (yptr[x] - minV) * 255 / (maxV - minV);
					m_img->setPixel(x, y, qRgb(val, val, val));
				}
			}
			break;
		case QImdebugType_INDEX_RGB:
			m_channels = 3;
			for (int i = w*h*3 - 1; i >= 0; i--)
			{
				maxV = std::max(maxV, data[i]);
				minV = std::min(minV, data[i]);
			}

			for (int y = 0; y < h; y++)
			{
				const T* yptr = data + (reverse_y ? h - 1 - y : y) * w * 3;
				for (int x = 0; x < w; x++)
				{
					float val[3];
					for (int k = 0; k < 3; k++)
					{
						if (maxV == minV)
							val[k] = 128;
						else
							val[k] = (yptr[x*3+k] - minV) * 255 / (maxV - minV);
					}
					m_img->setPixel(x, y, qRgb(val[0], val[1], val[2]));
				}
			}
			break;
		case QImdebugType_INDEX_RGBA:
			m_channels = 4;
			for (int i = 0; i < w*h*4; i+=4)
			{
				for (int k = 0; k < 3; k++)
				{
					maxV = std::max(maxV, data[i+k]);
					minV = std::min(minV, data[i+k]);
				}
			}

			for (int y = 0; y < h; y++)
			{
				const T* yptr = data + (reverse_y ? h - 1 - y : y) * w * 4;
				for (int x = 0; x < w; x++)
				{
					float val[4];
					for (int k = 0; k < 4; k++)
					{
						if (maxV == minV)
							val[k] = 128;
						else
							val[k] = (yptr[x * 4 + k] - minV) * 255 / (maxV - minV);
					}
					m_img->setPixel(x, y, qRgba(val[0], val[1], val[2], val[3]));
				}
			}
			break;
		default:
			break;
		}

		m_keptData.resize(w*h*m_channels*sizeof(T));
		memcpy(m_keptData.data(), data, m_keptData.size());

		return true;
	}
protected:
	inline void appendFormatData(QString& info, QPoint p)
	{
		if (p.x() < 0 || p.y() < 0 || p.x() >= m_img->width() || p.y() >= m_img->height())
		{
			info.append(QString().sprintf(" [out of range]"));
			return;
		}
		int pos = (p.x() + (m_reverse_y?m_img->height()-1-p.y():p.y())*m_img->width())*m_channels;
		switch (m_dataType)
		{
		case QImageDebugger_Byte:
			for (int k = 0; k < m_channels; k++)
			{
				uchar v = ((const uchar*)m_keptData.data())[pos + k];
				info.append(QString().sprintf(" %d", v));
			}
			break;
		case QImageDebugger_Int:
			for (int k = 0; k < m_channels; k++)
			{
				int v = ((const int*)m_keptData.data())[pos + k];
				info.append(QString().sprintf(" %d", v));
			}
			break;
		case QImageDebugger_Float:
			for (int k = 0; k < m_channels; k++)
			{
				float v = ((const float*)m_keptData.data())[pos + k];
				info.append(QString().sprintf(" %f", v));
			}
			break;
		default:
			break;
		}
	}

	virtual void paintEvent(QPaintEvent *ev)
	{
		QPainter painter(this);
		QRect geo(0, 0, width(), height());

		painter.setTransform(m_viewTransform);

		painter.drawImage(geo, *m_img);

		float S = sqrt(m_viewTransform.m11()*m_viewTransform.m22() - m_viewTransform.m12()*m_viewTransform.m21());
		if (S > 10)
		{
			float xb = 0, yb = 0;
			float xe = m_img->width();
			float ye = m_img->height();
			float sx = width() / float(m_img->width());
			float sy = height() / float(m_img->height());

			QPainterPath path;
			for (int y = 0; y < m_img->height(); y++)
			{
				path.moveTo(xb*sx, y*sy);
				path.lineTo(xe*sx, y*sy);
			}
			for (int x = 0; x < m_img->width(); x++)
			{
				path.moveTo(x*sx, yb*sy);
				path.lineTo(x*sx, ye*sy);
			}
			QPen pen(QColor(158, 158, 158));
			pen.setWidthF(1.f / S);
			painter.setPen(pen);
			painter.setBrush(Qt::NoBrush);
			painter.drawPath(path);
		}
	}

	virtual void keyReleaseEvent(QKeyEvent *)
	{

	}

	virtual void mousePressEvent(QMouseEvent *ev)
	{
		if (ev->buttons() == Qt::MiddleButton)
			m_viewTransform.reset();

		if (ev->buttons() == Qt::LeftButton)
		{
			m_mousePressPos = ev->pos();
			m_lastMousePos = m_mousePressPos;
		}
		update();
	}

	virtual void mouseReleaseEvent(QMouseEvent *)
	{

	}

	virtual void mouseDoubleClickEvent(QMouseEvent *)
	{

	}

	virtual void wheelEvent(QWheelEvent* ev)
	{
		float s = 1.2f;
		if (ev->delta() < 0)
			s = 1.f / s;

		QPointF fp = ev->pos();
		fp = m_viewTransform.inverted().map(fp);
		float S = sqrt(m_viewTransform.m11()*m_viewTransform.m22() - m_viewTransform.m12()*m_viewTransform.m21());
		float S1 = std::max(1.f/32.f, std::min(32.f, S*s));
		s = S1 / S;

		QPointF t(m_viewTransform.dx(), m_viewTransform.dy());
		m_viewTransform.translate(-t.x()/S, -t.y()/S);
		m_viewTransform.scale(s, s);
		QPointF t1 = ((S - S1)*fp + t)/S1;
		m_viewTransform.translate(t1.x(), t1.y());

		updateInfoByPos(ev->pos());

		update();
	}

	virtual void mouseMoveEvent(QMouseEvent * ev)
	{
		if (m_img == nullptr)
			return;

		setCursor(Qt::CrossCursor);

		if (ev->buttons() == Qt::LeftButton)
		{
			float S = sqrt(m_viewTransform.m11()*m_viewTransform.m22() - 
				m_viewTransform.m12()*m_viewTransform.m21());
			m_viewTransform.translate(
				(ev->pos().x() - m_lastMousePos.x()) / S,
				(ev->pos().y() - m_lastMousePos.y()) / S);
			m_lastMousePos = ev->pos();
			update();
		}
		
		updateInfoByPos(ev->pos());
	}

	void updateInfoByPos(QPointF fp)
	{
		fp = m_viewTransform.inverted().map(fp);

		fp.setX(fp.x() * m_img->width() / width());
		fp.setY(fp.y() * m_img->height() / height());

		QString info = m_name;
		if (!info.endsWith(":"))
			info.append(":");
		QPoint p(std::lroundf(fp.x()-0.5f), std::lroundf(fp.y()-0.5f));
		float S = sqrt(m_viewTransform.m11()*m_viewTransform.m22() - m_viewTransform.m12()*m_viewTransform.m21());
		info.append(QString().sprintf(" [%d%%](%d, %d):", std::lroundf(S*100), p.x(), p.y()));
		appendFormatData(info, p);
		setWindowTitle(info);
	}

	virtual void keyPressEvent(QKeyEvent * ev)
	{
		switch (ev->key())
		{
		default:
			break;
		case Qt::Key_S:
			if (ev->modifiers() & Qt::CTRL)
			{
				if (m_img)
				{
					QString name = QFileDialog::getSaveFileName(nullptr,
						"save image", "", "*.jpg;;*.png;;*.*");
					if (!name.isEmpty())
						m_img->save(name);
				}
			}
			break;
		}
	}

private:
	QImage* m_img;
	QString m_name;

	QImageDebugger_DataType m_dataType;
	QImdebugType m_type;
	std::vector<uchar> m_keptData;
	int m_channels;
	QTransform m_viewTransform;
	QPoint m_mousePressPos;
	QPoint m_lastMousePos;
	bool m_reverse_y;
};


class QImdebugFactory
{
public:
	QImdebugger* getNamedWindow(QString name)
	{
		QMap<QString, QImdebugger*>::iterator it = m_windows.find(name);
		if (it == m_windows.end())
		{
			QImdebugger* db = new QImdebugger(nullptr);
			db->setName(name);
			m_windows.insert(name, db);
			return db;
		}
		else
			return it.value();
	}
private:
	QMap<QString, QImdebugger*> m_windows;
};
QImdebugFactory g_qdebug_factory;


template<class T>
bool qimdebug(const char* name, const T* data, int w, int h, QImdebugType type, bool reverse_y)
{
	QImdebugger* debugger = g_qdebug_factory.getNamedWindow(name);
	assert(debugger);
	if (debugger == nullptr)
	{
		printf("create qt windows failed!\n");
		return false;
	}

	if (!debugger->setImage(data, w, h, type, reverse_y))
		return false;

	debugger->show();
	debugger->update();

	return true;
}

bool qimdebug(const char* name, const unsigned char* data, int w, int h, QImdebugType type, bool reverse_y)
{
	return qimdebug<unsigned char>(name, data, w, h, type, reverse_y);
}

bool qimdebug(const char* name, const float* data, int w, int h, QImdebugType type, bool reverse_y)
{
	return qimdebug<float>(name, data, w, h, type, reverse_y);
}

bool qimdebug(const char* name, const int* data, int w, int h, QImdebugType type, bool reverse_y)
{
	return qimdebug<int>(name, data, w, h, type, reverse_y);
}

