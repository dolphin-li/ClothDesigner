#ifndef CLOTHDESIGNER_H
#define CLOTHDESIGNER_H

#include <QtWidgets/QMainWindow>
#include "ui_clothdesigner.h"
#include <QSplitter>
#include <QPushButton>
#include <QSharedPointer>
#include <QSignalMapper>
#include "ldpMat\ldp_basic_mat.h"
#include "event_handles\Abstract3dEventHandle.h"
class Viewer3d;
class ClothDesigner : public QMainWindow
{
	Q_OBJECT

public:
	ClothDesigner(QWidget *parent = 0);
	~ClothDesigner();

	void timerEvent(QTimerEvent* ev);
private:
	Ui::ClothDesignerClass ui;
	QSplitter* m_splitter;
	QWidget* m_widget2d;
	Viewer3d* m_widget3d;
	int m_simulateTimer;
	int m_fpsTimer;
	//////////////////////////////////////////////////////////////////////////
protected:
	QMap<Abstract3dEventHandle::ProcessorType, QSharedPointer<QPushButton>> m_leftDockButtons;
	QSharedPointer<QSignalMapper> m_ldbSignalMapper;
	void initLeftDockActions();
	void addLeftDockWidgetButton(Abstract3dEventHandle::ProcessorType type);
	public slots:
	void leftDocButtonsClicked(int i);
};

#endif // CLOTHDESIGNER_H
