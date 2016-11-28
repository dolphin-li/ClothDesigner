/********************************************************************************
** Form generated from reading UI file 'clothdesigner.ui'
**
** Created by: Qt User Interface Compiler version 5.3.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_CLOTHDESIGNER_H
#define UI_CLOTHDESIGNER_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QDockWidget>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_ClothDesignerClass
{
public:
    QAction *actionLoad_project;
    QAction *actionOpen_body_mesh;
    QAction *actionImport_cloth_mesh;
    QAction *actionSave_project;
    QAction *actionLoad_svg;
    QWidget *centralWidget;
    QMenuBar *menuBar;
    QMenu *menuFile;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;
    QDockWidget *dockWidgetRight;
    QWidget *dockWidgetContentsRight;
    QGridLayout *gridLayout_2;
    QGroupBox *groupBox;
    QGridLayout *gridLayout;
    QLabel *label_13;
    QDoubleSpinBox *sbSparamStitchStiff;
    QLabel *label;
    QDoubleSpinBox *sbSparamSpringStiff;
    QLabel *label_10;
    QLabel *label_2;
    QDoubleSpinBox *sbSparamGravityX;
    QSpinBox *sbSparamTimeStepInv;
    QSpinBox *sbSparamLapDampIter;
    QLabel *label_3;
    QLabel *label_11;
    QLabel *label_4;
    QLabel *label_7;
    QLabel *label_9;
    QLabel *label_12;
    QDoubleSpinBox *sbSparamGravityY;
    QLabel *label_6;
    QLabel *label_5;
    QDoubleSpinBox *sbSparamUnderRelax;
    QDoubleSpinBox *sbSparamControlStiff;
    QSpinBox *sbSparamOuterIter;
    QDoubleSpinBox *sbSparamRho;
    QDoubleSpinBox *sbSparamBendStiff;
    QLabel *label_8;
    QSpinBox *sbSparamInnerIter;
    QDoubleSpinBox *sbSparamAirDamp;
    QDoubleSpinBox *sbSparamGravityZ;
    QDoubleSpinBox *sbSparamStitchSpeed;
    QSpacerItem *verticalSpacer;
    QPushButton *pbResetSimulation;
    QDockWidget *dockWidgetLower;
    QWidget *dockWidgetContentsLower;
    QDockWidget *dockWidgetLeft;
    QWidget *dockWidgetContentsLeft;

    void setupUi(QMainWindow *ClothDesignerClass)
    {
        if (ClothDesignerClass->objectName().isEmpty())
            ClothDesignerClass->setObjectName(QStringLiteral("ClothDesignerClass"));
        ClothDesignerClass->resize(1123, 881);
        actionLoad_project = new QAction(ClothDesignerClass);
        actionLoad_project->setObjectName(QStringLiteral("actionLoad_project"));
        actionOpen_body_mesh = new QAction(ClothDesignerClass);
        actionOpen_body_mesh->setObjectName(QStringLiteral("actionOpen_body_mesh"));
        actionImport_cloth_mesh = new QAction(ClothDesignerClass);
        actionImport_cloth_mesh->setObjectName(QStringLiteral("actionImport_cloth_mesh"));
        actionSave_project = new QAction(ClothDesignerClass);
        actionSave_project->setObjectName(QStringLiteral("actionSave_project"));
        actionLoad_svg = new QAction(ClothDesignerClass);
        actionLoad_svg->setObjectName(QStringLiteral("actionLoad_svg"));
        centralWidget = new QWidget(ClothDesignerClass);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        ClothDesignerClass->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(ClothDesignerClass);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 1123, 21));
        menuFile = new QMenu(menuBar);
        menuFile->setObjectName(QStringLiteral("menuFile"));
        ClothDesignerClass->setMenuBar(menuBar);
        mainToolBar = new QToolBar(ClothDesignerClass);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        ClothDesignerClass->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(ClothDesignerClass);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        ClothDesignerClass->setStatusBar(statusBar);
        dockWidgetRight = new QDockWidget(ClothDesignerClass);
        dockWidgetRight->setObjectName(QStringLiteral("dockWidgetRight"));
        dockWidgetRight->setMinimumSize(QSize(250, 459));
        dockWidgetRight->setMaximumSize(QSize(250, 524287));
        dockWidgetRight->setFeatures(QDockWidget::NoDockWidgetFeatures);
        dockWidgetContentsRight = new QWidget();
        dockWidgetContentsRight->setObjectName(QStringLiteral("dockWidgetContentsRight"));
        gridLayout_2 = new QGridLayout(dockWidgetContentsRight);
        gridLayout_2->setSpacing(6);
        gridLayout_2->setContentsMargins(11, 11, 11, 11);
        gridLayout_2->setObjectName(QStringLiteral("gridLayout_2"));
        groupBox = new QGroupBox(dockWidgetContentsRight);
        groupBox->setObjectName(QStringLiteral("groupBox"));
        gridLayout = new QGridLayout(groupBox);
        gridLayout->setSpacing(6);
        gridLayout->setContentsMargins(11, 11, 11, 11);
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        label_13 = new QLabel(groupBox);
        label_13->setObjectName(QStringLiteral("label_13"));

        gridLayout->addWidget(label_13, 11, 0, 1, 1);

        sbSparamStitchStiff = new QDoubleSpinBox(groupBox);
        sbSparamStitchStiff->setObjectName(QStringLiteral("sbSparamStitchStiff"));
        sbSparamStitchStiff->setMaximum(1e+08);

        gridLayout->addWidget(sbSparamStitchStiff, 10, 2, 1, 2);

        label = new QLabel(groupBox);
        label->setObjectName(QStringLiteral("label"));

        gridLayout->addWidget(label, 0, 0, 1, 1);

        sbSparamSpringStiff = new QDoubleSpinBox(groupBox);
        sbSparamSpringStiff->setObjectName(QStringLiteral("sbSparamSpringStiff"));
        sbSparamSpringStiff->setMaximum(1e+09);

        gridLayout->addWidget(sbSparamSpringStiff, 8, 2, 1, 2);

        label_10 = new QLabel(groupBox);
        label_10->setObjectName(QStringLiteral("label_10"));

        gridLayout->addWidget(label_10, 9, 0, 1, 1);

        label_2 = new QLabel(groupBox);
        label_2->setObjectName(QStringLiteral("label_2"));

        gridLayout->addWidget(label_2, 1, 0, 1, 1);

        sbSparamGravityX = new QDoubleSpinBox(groupBox);
        sbSparamGravityX->setObjectName(QStringLiteral("sbSparamGravityX"));
        sbSparamGravityX->setDecimals(1);
        sbSparamGravityX->setMinimum(-100);

        gridLayout->addWidget(sbSparamGravityX, 12, 1, 1, 1);

        sbSparamTimeStepInv = new QSpinBox(groupBox);
        sbSparamTimeStepInv->setObjectName(QStringLiteral("sbSparamTimeStepInv"));
        sbSparamTimeStepInv->setMaximum(9999);

        gridLayout->addWidget(sbSparamTimeStepInv, 2, 2, 1, 2);

        sbSparamLapDampIter = new QSpinBox(groupBox);
        sbSparamLapDampIter->setObjectName(QStringLiteral("sbSparamLapDampIter"));

        gridLayout->addWidget(sbSparamLapDampIter, 3, 2, 1, 2);

        label_3 = new QLabel(groupBox);
        label_3->setObjectName(QStringLiteral("label_3"));

        gridLayout->addWidget(label_3, 2, 0, 1, 1);

        label_11 = new QLabel(groupBox);
        label_11->setObjectName(QStringLiteral("label_11"));

        gridLayout->addWidget(label_11, 12, 0, 1, 1);

        label_4 = new QLabel(groupBox);
        label_4->setObjectName(QStringLiteral("label_4"));

        gridLayout->addWidget(label_4, 3, 0, 1, 2);

        label_7 = new QLabel(groupBox);
        label_7->setObjectName(QStringLiteral("label_7"));

        gridLayout->addWidget(label_7, 6, 0, 1, 1);

        label_9 = new QLabel(groupBox);
        label_9->setObjectName(QStringLiteral("label_9"));

        gridLayout->addWidget(label_9, 8, 0, 1, 1);

        label_12 = new QLabel(groupBox);
        label_12->setObjectName(QStringLiteral("label_12"));

        gridLayout->addWidget(label_12, 10, 0, 1, 1);

        sbSparamGravityY = new QDoubleSpinBox(groupBox);
        sbSparamGravityY->setObjectName(QStringLiteral("sbSparamGravityY"));
        sbSparamGravityY->setDecimals(1);
        sbSparamGravityY->setMinimum(-100);

        gridLayout->addWidget(sbSparamGravityY, 12, 2, 1, 1);

        label_6 = new QLabel(groupBox);
        label_6->setObjectName(QStringLiteral("label_6"));

        gridLayout->addWidget(label_6, 5, 0, 1, 1);

        label_5 = new QLabel(groupBox);
        label_5->setObjectName(QStringLiteral("label_5"));

        gridLayout->addWidget(label_5, 4, 0, 1, 1);

        sbSparamUnderRelax = new QDoubleSpinBox(groupBox);
        sbSparamUnderRelax->setObjectName(QStringLiteral("sbSparamUnderRelax"));
        sbSparamUnderRelax->setDecimals(5);
        sbSparamUnderRelax->setMaximum(1);

        gridLayout->addWidget(sbSparamUnderRelax, 7, 2, 1, 2);

        sbSparamControlStiff = new QDoubleSpinBox(groupBox);
        sbSparamControlStiff->setObjectName(QStringLiteral("sbSparamControlStiff"));
        sbSparamControlStiff->setMaximum(1e+09);

        gridLayout->addWidget(sbSparamControlStiff, 5, 2, 1, 2);

        sbSparamOuterIter = new QSpinBox(groupBox);
        sbSparamOuterIter->setObjectName(QStringLiteral("sbSparamOuterIter"));

        gridLayout->addWidget(sbSparamOuterIter, 0, 2, 1, 2);

        sbSparamRho = new QDoubleSpinBox(groupBox);
        sbSparamRho->setObjectName(QStringLiteral("sbSparamRho"));
        sbSparamRho->setDecimals(5);
        sbSparamRho->setMaximum(1);

        gridLayout->addWidget(sbSparamRho, 6, 2, 1, 2);

        sbSparamBendStiff = new QDoubleSpinBox(groupBox);
        sbSparamBendStiff->setObjectName(QStringLiteral("sbSparamBendStiff"));
        sbSparamBendStiff->setMaximum(9999);

        gridLayout->addWidget(sbSparamBendStiff, 9, 2, 1, 2);

        label_8 = new QLabel(groupBox);
        label_8->setObjectName(QStringLiteral("label_8"));

        gridLayout->addWidget(label_8, 7, 0, 1, 1);

        sbSparamInnerIter = new QSpinBox(groupBox);
        sbSparamInnerIter->setObjectName(QStringLiteral("sbSparamInnerIter"));
        sbSparamInnerIter->setMaximum(9999);

        gridLayout->addWidget(sbSparamInnerIter, 1, 2, 1, 2);

        sbSparamAirDamp = new QDoubleSpinBox(groupBox);
        sbSparamAirDamp->setObjectName(QStringLiteral("sbSparamAirDamp"));
        sbSparamAirDamp->setDecimals(5);
        sbSparamAirDamp->setMaximum(1);

        gridLayout->addWidget(sbSparamAirDamp, 4, 2, 1, 2);

        sbSparamGravityZ = new QDoubleSpinBox(groupBox);
        sbSparamGravityZ->setObjectName(QStringLiteral("sbSparamGravityZ"));
        sbSparamGravityZ->setDecimals(1);
        sbSparamGravityZ->setMinimum(-100);

        gridLayout->addWidget(sbSparamGravityZ, 12, 3, 1, 1);

        sbSparamStitchSpeed = new QDoubleSpinBox(groupBox);
        sbSparamStitchSpeed->setObjectName(QStringLiteral("sbSparamStitchSpeed"));
        sbSparamStitchSpeed->setDecimals(1);
        sbSparamStitchSpeed->setMaximum(100);

        gridLayout->addWidget(sbSparamStitchSpeed, 11, 2, 1, 2);


        gridLayout_2->addWidget(groupBox, 0, 0, 1, 1);

        verticalSpacer = new QSpacerItem(20, 309, QSizePolicy::Minimum, QSizePolicy::Expanding);

        gridLayout_2->addItem(verticalSpacer, 2, 0, 1, 1);

        pbResetSimulation = new QPushButton(dockWidgetContentsRight);
        pbResetSimulation->setObjectName(QStringLiteral("pbResetSimulation"));

        gridLayout_2->addWidget(pbResetSimulation, 1, 0, 1, 1);

        dockWidgetRight->setWidget(dockWidgetContentsRight);
        ClothDesignerClass->addDockWidget(static_cast<Qt::DockWidgetArea>(2), dockWidgetRight);
        dockWidgetLower = new QDockWidget(ClothDesignerClass);
        dockWidgetLower->setObjectName(QStringLiteral("dockWidgetLower"));
        dockWidgetLower->setMinimumSize(QSize(80, 150));
        dockWidgetLower->setFeatures(QDockWidget::NoDockWidgetFeatures);
        dockWidgetContentsLower = new QWidget();
        dockWidgetContentsLower->setObjectName(QStringLiteral("dockWidgetContentsLower"));
        dockWidgetLower->setWidget(dockWidgetContentsLower);
        ClothDesignerClass->addDockWidget(static_cast<Qt::DockWidgetArea>(8), dockWidgetLower);
        dockWidgetLeft = new QDockWidget(ClothDesignerClass);
        dockWidgetLeft->setObjectName(QStringLiteral("dockWidgetLeft"));
        dockWidgetLeft->setFeatures(QDockWidget::NoDockWidgetFeatures);
        dockWidgetLeft->setAllowedAreas(Qt::NoDockWidgetArea);
        dockWidgetContentsLeft = new QWidget();
        dockWidgetContentsLeft->setObjectName(QStringLiteral("dockWidgetContentsLeft"));
        dockWidgetLeft->setWidget(dockWidgetContentsLeft);
        ClothDesignerClass->addDockWidget(static_cast<Qt::DockWidgetArea>(1), dockWidgetLeft);

        menuBar->addAction(menuFile->menuAction());
        menuFile->addAction(actionLoad_project);
        menuFile->addAction(actionOpen_body_mesh);
        menuFile->addAction(actionImport_cloth_mesh);
        menuFile->addAction(actionSave_project);
        menuFile->addAction(actionLoad_svg);

        retranslateUi(ClothDesignerClass);

        QMetaObject::connectSlotsByName(ClothDesignerClass);
    } // setupUi

    void retranslateUi(QMainWindow *ClothDesignerClass)
    {
        ClothDesignerClass->setWindowTitle(QApplication::translate("ClothDesignerClass", "ClothDesigner", 0));
        actionLoad_project->setText(QApplication::translate("ClothDesignerClass", "open project", 0));
        actionOpen_body_mesh->setText(QApplication::translate("ClothDesignerClass", "import body mesh", 0));
        actionOpen_body_mesh->setShortcut(QApplication::translate("ClothDesignerClass", "Ctrl+O", 0));
        actionImport_cloth_mesh->setText(QApplication::translate("ClothDesignerClass", "import cloth mesh", 0));
        actionSave_project->setText(QApplication::translate("ClothDesignerClass", "save project", 0));
        actionSave_project->setShortcut(QApplication::translate("ClothDesignerClass", "Ctrl+S", 0));
        actionLoad_svg->setText(QApplication::translate("ClothDesignerClass", "load svg", 0));
        menuFile->setTitle(QApplication::translate("ClothDesignerClass", "file", 0));
        groupBox->setTitle(QApplication::translate("ClothDesignerClass", "Simulation Param", 0));
        label_13->setText(QApplication::translate("ClothDesignerClass", "stitch speed", 0));
        label->setText(QApplication::translate("ClothDesignerClass", "outer iter", 0));
        label_10->setText(QApplication::translate("ClothDesignerClass", "bend stiff", 0));
        label_2->setText(QApplication::translate("ClothDesignerClass", "inner iter", 0));
        label_3->setText(QApplication::translate("ClothDesignerClass", "time step inv", 0));
        label_11->setText(QApplication::translate("ClothDesignerClass", "gravity", 0));
        label_4->setText(QApplication::translate("ClothDesignerClass", "lap damp iter", 0));
        label_7->setText(QApplication::translate("ClothDesignerClass", "rho", 0));
        label_9->setText(QApplication::translate("ClothDesignerClass", "spring stiff", 0));
        label_12->setText(QApplication::translate("ClothDesignerClass", "stitch stiff", 0));
        label_6->setText(QApplication::translate("ClothDesignerClass", "control stiff", 0));
        label_5->setText(QApplication::translate("ClothDesignerClass", "air damp", 0));
        label_8->setText(QApplication::translate("ClothDesignerClass", "under relax", 0));
        pbResetSimulation->setText(QApplication::translate("ClothDesignerClass", "reset simulaton", 0));
        pbResetSimulation->setShortcut(QApplication::translate("ClothDesignerClass", "1", 0));
    } // retranslateUi

};

namespace Ui {
    class ClothDesignerClass: public Ui_ClothDesignerClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_CLOTHDESIGNER_H
