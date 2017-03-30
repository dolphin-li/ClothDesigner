/********************************************************************************
** Form generated from reading UI file 'ArcSimWindow.ui'
**
** Created by: Qt User Interface Compiler version 5.3.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_ARCSIMWINDOW_H
#define UI_ARCSIMWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QDockWidget>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_ArcsimWindow
{
public:
    QAction *actionLoad_conf;
    QAction *actionSave_cloth;
    QWidget *centralwidget;
    QGridLayout *gridLayout;
    QWidget *widget;
    QMenuBar *menubar;
    QMenu *menuFile;
    QStatusBar *statusbar;
    QDockWidget *dockWidget;
    QWidget *dockWidgetContents_2;

    void setupUi(QMainWindow *ArcsimWindow)
    {
        if (ArcsimWindow->objectName().isEmpty())
            ArcsimWindow->setObjectName(QStringLiteral("ArcsimWindow"));
        ArcsimWindow->resize(1083, 856);
        actionLoad_conf = new QAction(ArcsimWindow);
        actionLoad_conf->setObjectName(QStringLiteral("actionLoad_conf"));
        actionSave_cloth = new QAction(ArcsimWindow);
        actionSave_cloth->setObjectName(QStringLiteral("actionSave_cloth"));
        centralwidget = new QWidget(ArcsimWindow);
        centralwidget->setObjectName(QStringLiteral("centralwidget"));
        gridLayout = new QGridLayout(centralwidget);
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        widget = new QWidget(centralwidget);
        widget->setObjectName(QStringLiteral("widget"));

        gridLayout->addWidget(widget, 0, 0, 1, 1);

        ArcsimWindow->setCentralWidget(centralwidget);
        menubar = new QMenuBar(ArcsimWindow);
        menubar->setObjectName(QStringLiteral("menubar"));
        menubar->setGeometry(QRect(0, 0, 1083, 23));
        menuFile = new QMenu(menubar);
        menuFile->setObjectName(QStringLiteral("menuFile"));
        ArcsimWindow->setMenuBar(menubar);
        statusbar = new QStatusBar(ArcsimWindow);
        statusbar->setObjectName(QStringLiteral("statusbar"));
        ArcsimWindow->setStatusBar(statusbar);
        dockWidget = new QDockWidget(ArcsimWindow);
        dockWidget->setObjectName(QStringLiteral("dockWidget"));
        dockWidget->setMinimumSize(QSize(150, 38));
        dockWidgetContents_2 = new QWidget();
        dockWidgetContents_2->setObjectName(QStringLiteral("dockWidgetContents_2"));
        dockWidget->setWidget(dockWidgetContents_2);
        ArcsimWindow->addDockWidget(static_cast<Qt::DockWidgetArea>(1), dockWidget);

        menubar->addAction(menuFile->menuAction());
        menuFile->addAction(actionLoad_conf);
        menuFile->addAction(actionSave_cloth);

        retranslateUi(ArcsimWindow);

        QMetaObject::connectSlotsByName(ArcsimWindow);
    } // setupUi

    void retranslateUi(QMainWindow *ArcsimWindow)
    {
        ArcsimWindow->setWindowTitle(QApplication::translate("ArcsimWindow", "MainWindow", 0));
        actionLoad_conf->setText(QApplication::translate("ArcsimWindow", "load conf", 0));
        actionSave_cloth->setText(QApplication::translate("ArcsimWindow", "save cloth", 0));
        menuFile->setTitle(QApplication::translate("ArcsimWindow", "file", 0));
    } // retranslateUi

};

namespace Ui {
    class ArcsimWindow: public Ui_ArcsimWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_ARCSIMWINDOW_H
