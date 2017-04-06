#include "gl\glew.h"
#include "clothdesigner.h"
#include <QtWidgets/QApplication>
#include <QFile>
#include <QTextStream>
#include <GL\glut.h>

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);

	glewInit();
	glutInit(&argc, argv);

	QFile f(":qdarkstyle/style.qss");
	if (!f.exists())
	{
		printf("Unable to set stylesheet, file not found\n");
	}
	else
	{
		f.open(QFile::ReadOnly | QFile::Text);
		QTextStream ts(&f);
		qApp->setStyleSheet(ts.readAll());
	}

	ClothDesigner w;
	w.showMaximized();
	return a.exec();
}
