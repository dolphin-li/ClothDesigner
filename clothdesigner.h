#ifndef CLOTHDESIGNER_H
#define CLOTHDESIGNER_H

#include <QtWidgets/QMainWindow>
#include "ui_clothdesigner.h"
#include <QSplitter>
class Viewer3d;
class ClothDesigner : public QMainWindow
{
	Q_OBJECT

public:
	ClothDesigner(QWidget *parent = 0);
	~ClothDesigner();

private:
	Ui::ClothDesignerClass ui;
	QSplitter* m_splitter;
	QWidget* m_widget2d;
	Viewer3d* m_widget3d;
};

#endif // CLOTHDESIGNER_H
