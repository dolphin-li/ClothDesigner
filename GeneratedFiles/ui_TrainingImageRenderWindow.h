/********************************************************************************
** Form generated from reading UI file 'TrainingImageRenderWindow.ui'
**
** Created by: Qt User Interface Compiler version 5.3.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_TRAININGIMAGERENDERWINDOW_H
#define UI_TRAININGIMAGERENDERWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QWidget>
#include "trainingimagerenderview.h"

QT_BEGIN_NAMESPACE

class Ui_TrainingImageRenderWindow
{
public:
    QAction *actionLoad_cloth_mesh;
    QAction *actionBatch_render_dist_map;
    QAction *actionRender_current_to_distmap;
    QWidget *centralwidget;
    QGridLayout *gridLayout;
    TrainingImageRenderView *widget;
    QSpacerItem *horizontalSpacer;
    QSpacerItem *horizontalSpacer_2;
    QMenuBar *menubar;
    QMenu *menuFile;
    QMenu *menuRender;
    QStatusBar *statusbar;

    void setupUi(QMainWindow *TrainingImageRenderWindow)
    {
        if (TrainingImageRenderWindow->objectName().isEmpty())
            TrainingImageRenderWindow->setObjectName(QStringLiteral("TrainingImageRenderWindow"));
        TrainingImageRenderWindow->resize(992, 843);
        actionLoad_cloth_mesh = new QAction(TrainingImageRenderWindow);
        actionLoad_cloth_mesh->setObjectName(QStringLiteral("actionLoad_cloth_mesh"));
        actionBatch_render_dist_map = new QAction(TrainingImageRenderWindow);
        actionBatch_render_dist_map->setObjectName(QStringLiteral("actionBatch_render_dist_map"));
        actionRender_current_to_distmap = new QAction(TrainingImageRenderWindow);
        actionRender_current_to_distmap->setObjectName(QStringLiteral("actionRender_current_to_distmap"));
        centralwidget = new QWidget(TrainingImageRenderWindow);
        centralwidget->setObjectName(QStringLiteral("centralwidget"));
        gridLayout = new QGridLayout(centralwidget);
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        widget = new TrainingImageRenderView(centralwidget);
        widget->setObjectName(QStringLiteral("widget"));

        gridLayout->addWidget(widget, 0, 1, 1, 1);

        horizontalSpacer = new QSpacerItem(150, 20, QSizePolicy::Fixed, QSizePolicy::Minimum);

        gridLayout->addItem(horizontalSpacer, 0, 0, 1, 1);

        horizontalSpacer_2 = new QSpacerItem(150, 20, QSizePolicy::Fixed, QSizePolicy::Minimum);

        gridLayout->addItem(horizontalSpacer_2, 0, 2, 1, 1);

        TrainingImageRenderWindow->setCentralWidget(centralwidget);
        menubar = new QMenuBar(TrainingImageRenderWindow);
        menubar->setObjectName(QStringLiteral("menubar"));
        menubar->setGeometry(QRect(0, 0, 992, 21));
        menuFile = new QMenu(menubar);
        menuFile->setObjectName(QStringLiteral("menuFile"));
        menuRender = new QMenu(menubar);
        menuRender->setObjectName(QStringLiteral("menuRender"));
        TrainingImageRenderWindow->setMenuBar(menubar);
        statusbar = new QStatusBar(TrainingImageRenderWindow);
        statusbar->setObjectName(QStringLiteral("statusbar"));
        TrainingImageRenderWindow->setStatusBar(statusbar);

        menubar->addAction(menuFile->menuAction());
        menubar->addAction(menuRender->menuAction());
        menuFile->addAction(actionLoad_cloth_mesh);
        menuFile->addAction(actionBatch_render_dist_map);
        menuRender->addAction(actionRender_current_to_distmap);

        retranslateUi(TrainingImageRenderWindow);

        QMetaObject::connectSlotsByName(TrainingImageRenderWindow);
    } // setupUi

    void retranslateUi(QMainWindow *TrainingImageRenderWindow)
    {
        TrainingImageRenderWindow->setWindowTitle(QApplication::translate("TrainingImageRenderWindow", "Training Image Render Window", 0));
        actionLoad_cloth_mesh->setText(QApplication::translate("TrainingImageRenderWindow", "load cloth mesh", 0));
        actionBatch_render_dist_map->setText(QApplication::translate("TrainingImageRenderWindow", "batch render dist map", 0));
        actionRender_current_to_distmap->setText(QApplication::translate("TrainingImageRenderWindow", "render current to distmap", 0));
        menuFile->setTitle(QApplication::translate("TrainingImageRenderWindow", "file", 0));
        menuRender->setTitle(QApplication::translate("TrainingImageRenderWindow", "render", 0));
    } // retranslateUi

};

namespace Ui {
    class TrainingImageRenderWindow: public Ui_TrainingImageRenderWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_TRAININGIMAGERENDERWINDOW_H
