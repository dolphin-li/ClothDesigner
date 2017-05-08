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
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QComboBox>
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
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_ClothDesignerClass
{
public:
    QAction *actionLoad_project;
    QAction *actionImport_body_mesh;
    QAction *actionImport_cloth_mesh;
    QAction *actionSave_project;
    QAction *actionLoad_svg;
    QAction *actionPrev;
    QAction *actionNext;
    QAction *actionPlace_3d_by_2d;
    QAction *actionExport_body_mesh;
    QAction *actionExport_cloth_mesh;
    QAction *actionSave_as;
    QAction *actionExport_batch_simulation;
    QAction *actionTraining_image_render;
    QAction *actionArcsim;
    QWidget *centralWidget;
    QMenuBar *menuBar;
    QMenu *menuFile;
    QMenu *menuEdit;
    QMenu *menuHistory;
    QMenu *menuWindows;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;
    QDockWidget *dockWidgetRight;
    QWidget *dockWidgetContentsRight;
    QGridLayout *gridLayout_2;
    QTabWidget *tabWidget;
    QWidget *tabSimulation;
    QGridLayout *gridLayout_5;
    QGroupBox *groupBox;
    QGridLayout *gridLayout;
    QDoubleSpinBox *sbSparamGravityY;
    QLabel *label_11;
    QDoubleSpinBox *sbSparamGravityZ;
    QLabel *label_9;
    QCheckBox *cbSelfCollision;
    QDoubleSpinBox *sbSparamBendStiff;
    QDoubleSpinBox *sbSparamGravityX;
    QLabel *label_10;
    QDoubleSpinBox *sbSparamSpringStiff;
    QLabel *label_15;
    QDoubleSpinBox *sbDparamTriangleSize;
    QGroupBox *groupBox_2;
    QGridLayout *gridLayout_6;
    QLabel *label_16;
    QDoubleSpinBox *dbPieceSpringMult;
    QDoubleSpinBox *dbPieceBendMult;
    QLabel *label_17;
    QLabel *label_18;
    QComboBox *cbPieceMaterialName;
    QSpacerItem *verticalSpacer;
    QGroupBox *groupBox_3;
    QGridLayout *gridLayout_3;
    QLabel *label;
    QSpinBox *sbSewParamAngle;
    QWidget *tab;
    QGridLayout *gridLayout_7;
    QPushButton *pbSaveSmplCoeffs;
    QSpacerItem *verticalSpacer_3;
    QPushButton *pbLoadSmplCoeffs;
    QPushButton *pbResetSmplCoeffs;
    QGroupBox *gpSmplBodyCoeffs;
    QPushButton *pbBindClothesToSmpl;
    QPushButton *pbLoadSmplFromXml;
    QDockWidget *dockWidgetLower;
    QWidget *dockWidgetContentsLower;
    QGridLayout *gridLayout_4;
    QSpacerItem *horizontalSpacer;
    QPushButton *pbFlipPolygon;
    QPushButton *pbResetSimulation;
    QPushButton *pbMirrorSelected;
    QPushButton *pbCopySelected;

    void setupUi(QMainWindow *ClothDesignerClass)
    {
        if (ClothDesignerClass->objectName().isEmpty())
            ClothDesignerClass->setObjectName(QStringLiteral("ClothDesignerClass"));
        ClothDesignerClass->resize(890, 784);
        ClothDesignerClass->setMinimumSize(QSize(0, 0));
        actionLoad_project = new QAction(ClothDesignerClass);
        actionLoad_project->setObjectName(QStringLiteral("actionLoad_project"));
        actionImport_body_mesh = new QAction(ClothDesignerClass);
        actionImport_body_mesh->setObjectName(QStringLiteral("actionImport_body_mesh"));
        actionImport_cloth_mesh = new QAction(ClothDesignerClass);
        actionImport_cloth_mesh->setObjectName(QStringLiteral("actionImport_cloth_mesh"));
        actionSave_project = new QAction(ClothDesignerClass);
        actionSave_project->setObjectName(QStringLiteral("actionSave_project"));
        actionLoad_svg = new QAction(ClothDesignerClass);
        actionLoad_svg->setObjectName(QStringLiteral("actionLoad_svg"));
        actionPrev = new QAction(ClothDesignerClass);
        actionPrev->setObjectName(QStringLiteral("actionPrev"));
        actionNext = new QAction(ClothDesignerClass);
        actionNext->setObjectName(QStringLiteral("actionNext"));
        actionPlace_3d_by_2d = new QAction(ClothDesignerClass);
        actionPlace_3d_by_2d->setObjectName(QStringLiteral("actionPlace_3d_by_2d"));
        actionExport_body_mesh = new QAction(ClothDesignerClass);
        actionExport_body_mesh->setObjectName(QStringLiteral("actionExport_body_mesh"));
        actionExport_cloth_mesh = new QAction(ClothDesignerClass);
        actionExport_cloth_mesh->setObjectName(QStringLiteral("actionExport_cloth_mesh"));
        actionSave_as = new QAction(ClothDesignerClass);
        actionSave_as->setObjectName(QStringLiteral("actionSave_as"));
        actionExport_batch_simulation = new QAction(ClothDesignerClass);
        actionExport_batch_simulation->setObjectName(QStringLiteral("actionExport_batch_simulation"));
        actionTraining_image_render = new QAction(ClothDesignerClass);
        actionTraining_image_render->setObjectName(QStringLiteral("actionTraining_image_render"));
        actionArcsim = new QAction(ClothDesignerClass);
        actionArcsim->setObjectName(QStringLiteral("actionArcsim"));
        centralWidget = new QWidget(ClothDesignerClass);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        ClothDesignerClass->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(ClothDesignerClass);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 890, 23));
        menuFile = new QMenu(menuBar);
        menuFile->setObjectName(QStringLiteral("menuFile"));
        menuEdit = new QMenu(menuBar);
        menuEdit->setObjectName(QStringLiteral("menuEdit"));
        menuHistory = new QMenu(menuBar);
        menuHistory->setObjectName(QStringLiteral("menuHistory"));
        menuWindows = new QMenu(menuBar);
        menuWindows->setObjectName(QStringLiteral("menuWindows"));
        ClothDesignerClass->setMenuBar(menuBar);
        mainToolBar = new QToolBar(ClothDesignerClass);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        ClothDesignerClass->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(ClothDesignerClass);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        ClothDesignerClass->setStatusBar(statusBar);
        dockWidgetRight = new QDockWidget(ClothDesignerClass);
        dockWidgetRight->setObjectName(QStringLiteral("dockWidgetRight"));
        dockWidgetRight->setMinimumSize(QSize(300, 505));
        dockWidgetRight->setMaximumSize(QSize(300, 524287));
        dockWidgetRight->setFeatures(QDockWidget::NoDockWidgetFeatures);
        dockWidgetContentsRight = new QWidget();
        dockWidgetContentsRight->setObjectName(QStringLiteral("dockWidgetContentsRight"));
        gridLayout_2 = new QGridLayout(dockWidgetContentsRight);
        gridLayout_2->setSpacing(6);
        gridLayout_2->setContentsMargins(11, 11, 11, 11);
        gridLayout_2->setObjectName(QStringLiteral("gridLayout_2"));
        tabWidget = new QTabWidget(dockWidgetContentsRight);
        tabWidget->setObjectName(QStringLiteral("tabWidget"));
        tabSimulation = new QWidget();
        tabSimulation->setObjectName(QStringLiteral("tabSimulation"));
        gridLayout_5 = new QGridLayout(tabSimulation);
        gridLayout_5->setSpacing(6);
        gridLayout_5->setContentsMargins(11, 11, 11, 11);
        gridLayout_5->setObjectName(QStringLiteral("gridLayout_5"));
        groupBox = new QGroupBox(tabSimulation);
        groupBox->setObjectName(QStringLiteral("groupBox"));
        gridLayout = new QGridLayout(groupBox);
        gridLayout->setSpacing(6);
        gridLayout->setContentsMargins(11, 11, 11, 11);
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        sbSparamGravityY = new QDoubleSpinBox(groupBox);
        sbSparamGravityY->setObjectName(QStringLiteral("sbSparamGravityY"));
        sbSparamGravityY->setDecimals(1);
        sbSparamGravityY->setMinimum(-100);

        gridLayout->addWidget(sbSparamGravityY, 3, 2, 1, 1);

        label_11 = new QLabel(groupBox);
        label_11->setObjectName(QStringLiteral("label_11"));

        gridLayout->addWidget(label_11, 3, 0, 1, 1);

        sbSparamGravityZ = new QDoubleSpinBox(groupBox);
        sbSparamGravityZ->setObjectName(QStringLiteral("sbSparamGravityZ"));
        sbSparamGravityZ->setDecimals(1);
        sbSparamGravityZ->setMinimum(-100);

        gridLayout->addWidget(sbSparamGravityZ, 3, 3, 1, 1);

        label_9 = new QLabel(groupBox);
        label_9->setObjectName(QStringLiteral("label_9"));

        gridLayout->addWidget(label_9, 0, 0, 1, 2);

        cbSelfCollision = new QCheckBox(groupBox);
        cbSelfCollision->setObjectName(QStringLiteral("cbSelfCollision"));

        gridLayout->addWidget(cbSelfCollision, 5, 0, 1, 2);

        sbSparamBendStiff = new QDoubleSpinBox(groupBox);
        sbSparamBendStiff->setObjectName(QStringLiteral("sbSparamBendStiff"));
        sbSparamBendStiff->setMaximum(9999);

        gridLayout->addWidget(sbSparamBendStiff, 1, 2, 1, 2);

        sbSparamGravityX = new QDoubleSpinBox(groupBox);
        sbSparamGravityX->setObjectName(QStringLiteral("sbSparamGravityX"));
        sbSparamGravityX->setDecimals(1);
        sbSparamGravityX->setMinimum(-100);

        gridLayout->addWidget(sbSparamGravityX, 3, 1, 1, 1);

        label_10 = new QLabel(groupBox);
        label_10->setObjectName(QStringLiteral("label_10"));

        gridLayout->addWidget(label_10, 1, 0, 1, 2);

        sbSparamSpringStiff = new QDoubleSpinBox(groupBox);
        sbSparamSpringStiff->setObjectName(QStringLiteral("sbSparamSpringStiff"));
        sbSparamSpringStiff->setMaximum(1e+09);

        gridLayout->addWidget(sbSparamSpringStiff, 0, 2, 1, 2);

        label_15 = new QLabel(groupBox);
        label_15->setObjectName(QStringLiteral("label_15"));
        label_15->setMinimumSize(QSize(0, 25));

        gridLayout->addWidget(label_15, 4, 0, 1, 2);

        sbDparamTriangleSize = new QDoubleSpinBox(groupBox);
        sbDparamTriangleSize->setObjectName(QStringLiteral("sbDparamTriangleSize"));
        sbDparamTriangleSize->setMinimumSize(QSize(0, 25));
        sbDparamTriangleSize->setDecimals(0);
        sbDparamTriangleSize->setMinimum(1);
        sbDparamTriangleSize->setMaximum(1000);

        gridLayout->addWidget(sbDparamTriangleSize, 4, 2, 1, 2);


        gridLayout_5->addWidget(groupBox, 1, 0, 1, 1);

        groupBox_2 = new QGroupBox(tabSimulation);
        groupBox_2->setObjectName(QStringLiteral("groupBox_2"));
        gridLayout_6 = new QGridLayout(groupBox_2);
        gridLayout_6->setSpacing(6);
        gridLayout_6->setContentsMargins(11, 11, 11, 11);
        gridLayout_6->setObjectName(QStringLiteral("gridLayout_6"));
        label_16 = new QLabel(groupBox_2);
        label_16->setObjectName(QStringLiteral("label_16"));

        gridLayout_6->addWidget(label_16, 0, 0, 1, 1);

        dbPieceSpringMult = new QDoubleSpinBox(groupBox_2);
        dbPieceSpringMult->setObjectName(QStringLiteral("dbPieceSpringMult"));
        dbPieceSpringMult->setDecimals(4);
        dbPieceSpringMult->setMaximum(999999);

        gridLayout_6->addWidget(dbPieceSpringMult, 1, 1, 1, 1);

        dbPieceBendMult = new QDoubleSpinBox(groupBox_2);
        dbPieceBendMult->setObjectName(QStringLiteral("dbPieceBendMult"));
        dbPieceBendMult->setDecimals(4);
        dbPieceBendMult->setMaximum(999999);

        gridLayout_6->addWidget(dbPieceBendMult, 0, 1, 1, 1);

        label_17 = new QLabel(groupBox_2);
        label_17->setObjectName(QStringLiteral("label_17"));

        gridLayout_6->addWidget(label_17, 1, 0, 1, 1);

        label_18 = new QLabel(groupBox_2);
        label_18->setObjectName(QStringLiteral("label_18"));

        gridLayout_6->addWidget(label_18, 2, 0, 1, 1);

        cbPieceMaterialName = new QComboBox(groupBox_2);
        cbPieceMaterialName->setObjectName(QStringLiteral("cbPieceMaterialName"));

        gridLayout_6->addWidget(cbPieceMaterialName, 2, 1, 1, 1);


        gridLayout_5->addWidget(groupBox_2, 4, 0, 1, 1);

        verticalSpacer = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        gridLayout_5->addItem(verticalSpacer, 6, 0, 1, 1);

        groupBox_3 = new QGroupBox(tabSimulation);
        groupBox_3->setObjectName(QStringLiteral("groupBox_3"));
        groupBox_3->setMinimumSize(QSize(0, 0));
        gridLayout_3 = new QGridLayout(groupBox_3);
        gridLayout_3->setSpacing(6);
        gridLayout_3->setContentsMargins(11, 11, 11, 11);
        gridLayout_3->setObjectName(QStringLiteral("gridLayout_3"));
        label = new QLabel(groupBox_3);
        label->setObjectName(QStringLiteral("label"));

        gridLayout_3->addWidget(label, 0, 0, 1, 1);

        sbSewParamAngle = new QSpinBox(groupBox_3);
        sbSewParamAngle->setObjectName(QStringLiteral("sbSewParamAngle"));
        sbSewParamAngle->setMinimum(-180);
        sbSewParamAngle->setMaximum(180);

        gridLayout_3->addWidget(sbSewParamAngle, 0, 1, 1, 1);


        gridLayout_5->addWidget(groupBox_3, 5, 0, 1, 1);

        tabWidget->addTab(tabSimulation, QString());
        tab = new QWidget();
        tab->setObjectName(QStringLiteral("tab"));
        gridLayout_7 = new QGridLayout(tab);
        gridLayout_7->setSpacing(6);
        gridLayout_7->setContentsMargins(11, 11, 11, 11);
        gridLayout_7->setObjectName(QStringLiteral("gridLayout_7"));
        pbSaveSmplCoeffs = new QPushButton(tab);
        pbSaveSmplCoeffs->setObjectName(QStringLiteral("pbSaveSmplCoeffs"));

        gridLayout_7->addWidget(pbSaveSmplCoeffs, 2, 0, 1, 1);

        verticalSpacer_3 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        gridLayout_7->addItem(verticalSpacer_3, 4, 0, 1, 2);

        pbLoadSmplCoeffs = new QPushButton(tab);
        pbLoadSmplCoeffs->setObjectName(QStringLiteral("pbLoadSmplCoeffs"));

        gridLayout_7->addWidget(pbLoadSmplCoeffs, 2, 1, 1, 1);

        pbResetSmplCoeffs = new QPushButton(tab);
        pbResetSmplCoeffs->setObjectName(QStringLiteral("pbResetSmplCoeffs"));

        gridLayout_7->addWidget(pbResetSmplCoeffs, 1, 0, 1, 1);

        gpSmplBodyCoeffs = new QGroupBox(tab);
        gpSmplBodyCoeffs->setObjectName(QStringLiteral("gpSmplBodyCoeffs"));

        gridLayout_7->addWidget(gpSmplBodyCoeffs, 0, 0, 1, 2);

        pbBindClothesToSmpl = new QPushButton(tab);
        pbBindClothesToSmpl->setObjectName(QStringLiteral("pbBindClothesToSmpl"));

        gridLayout_7->addWidget(pbBindClothesToSmpl, 3, 0, 1, 1);

        pbLoadSmplFromXml = new QPushButton(tab);
        pbLoadSmplFromXml->setObjectName(QStringLiteral("pbLoadSmplFromXml"));

        gridLayout_7->addWidget(pbLoadSmplFromXml, 3, 1, 1, 1);

        tabWidget->addTab(tab, QString());

        gridLayout_2->addWidget(tabWidget, 0, 0, 1, 1);

        dockWidgetRight->setWidget(dockWidgetContentsRight);
        ClothDesignerClass->addDockWidget(static_cast<Qt::DockWidgetArea>(2), dockWidgetRight);
        dockWidgetLower = new QDockWidget(ClothDesignerClass);
        dockWidgetLower->setObjectName(QStringLiteral("dockWidgetLower"));
        dockWidgetLower->setMinimumSize(QSize(270, 156));
        dockWidgetLower->setFeatures(QDockWidget::NoDockWidgetFeatures);
        dockWidgetContentsLower = new QWidget();
        dockWidgetContentsLower->setObjectName(QStringLiteral("dockWidgetContentsLower"));
        gridLayout_4 = new QGridLayout(dockWidgetContentsLower);
        gridLayout_4->setSpacing(6);
        gridLayout_4->setContentsMargins(11, 11, 11, 11);
        gridLayout_4->setObjectName(QStringLiteral("gridLayout_4"));
        horizontalSpacer = new QSpacerItem(937, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        gridLayout_4->addItem(horizontalSpacer, 0, 1, 3, 1);

        pbFlipPolygon = new QPushButton(dockWidgetContentsLower);
        pbFlipPolygon->setObjectName(QStringLiteral("pbFlipPolygon"));
        pbFlipPolygon->setMinimumSize(QSize(0, 25));

        gridLayout_4->addWidget(pbFlipPolygon, 1, 0, 1, 1);

        pbResetSimulation = new QPushButton(dockWidgetContentsLower);
        pbResetSimulation->setObjectName(QStringLiteral("pbResetSimulation"));

        gridLayout_4->addWidget(pbResetSimulation, 0, 0, 1, 1);

        pbMirrorSelected = new QPushButton(dockWidgetContentsLower);
        pbMirrorSelected->setObjectName(QStringLiteral("pbMirrorSelected"));
        pbMirrorSelected->setMinimumSize(QSize(0, 25));

        gridLayout_4->addWidget(pbMirrorSelected, 2, 0, 1, 1);

        pbCopySelected = new QPushButton(dockWidgetContentsLower);
        pbCopySelected->setObjectName(QStringLiteral("pbCopySelected"));
        pbCopySelected->setMinimumSize(QSize(0, 25));

        gridLayout_4->addWidget(pbCopySelected, 3, 0, 1, 1);

        dockWidgetLower->setWidget(dockWidgetContentsLower);
        ClothDesignerClass->addDockWidget(static_cast<Qt::DockWidgetArea>(8), dockWidgetLower);

        menuBar->addAction(menuFile->menuAction());
        menuBar->addAction(menuEdit->menuAction());
        menuBar->addAction(menuHistory->menuAction());
        menuBar->addAction(menuWindows->menuAction());
        menuFile->addAction(actionLoad_project);
        menuFile->addAction(actionSave_project);
        menuFile->addAction(actionSave_as);
        menuFile->addAction(actionLoad_svg);
        menuFile->addSeparator();
        menuFile->addAction(actionExport_body_mesh);
        menuFile->addAction(actionExport_cloth_mesh);
        menuFile->addAction(actionExport_batch_simulation);
        menuEdit->addAction(actionPlace_3d_by_2d);
        menuHistory->addAction(actionPrev);
        menuHistory->addAction(actionNext);
        menuWindows->addAction(actionTraining_image_render);
        menuWindows->addAction(actionArcsim);

        retranslateUi(ClothDesignerClass);

        tabWidget->setCurrentIndex(0);


        QMetaObject::connectSlotsByName(ClothDesignerClass);
    } // setupUi

    void retranslateUi(QMainWindow *ClothDesignerClass)
    {
        ClothDesignerClass->setWindowTitle(QApplication::translate("ClothDesignerClass", "ClothDesigner", 0));
        actionLoad_project->setText(QApplication::translate("ClothDesignerClass", "open project", 0));
        actionLoad_project->setShortcut(QApplication::translate("ClothDesignerClass", "Ctrl+O", 0));
        actionImport_body_mesh->setText(QApplication::translate("ClothDesignerClass", "import body mesh", 0));
        actionImport_cloth_mesh->setText(QApplication::translate("ClothDesignerClass", "import cloth mesh", 0));
        actionSave_project->setText(QApplication::translate("ClothDesignerClass", "save project", 0));
        actionSave_project->setShortcut(QApplication::translate("ClothDesignerClass", "Ctrl+S", 0));
        actionLoad_svg->setText(QApplication::translate("ClothDesignerClass", "load svg", 0));
        actionPrev->setText(QApplication::translate("ClothDesignerClass", "prev", 0));
        actionPrev->setShortcut(QApplication::translate("ClothDesignerClass", "Ctrl+Z", 0));
        actionNext->setText(QApplication::translate("ClothDesignerClass", "next", 0));
        actionNext->setShortcut(QApplication::translate("ClothDesignerClass", "Ctrl+Shift+Z", 0));
        actionPlace_3d_by_2d->setText(QApplication::translate("ClothDesignerClass", "place 3d by 2d", 0));
        actionExport_body_mesh->setText(QApplication::translate("ClothDesignerClass", "export body mesh", 0));
        actionExport_cloth_mesh->setText(QApplication::translate("ClothDesignerClass", "export cloth mesh", 0));
        actionSave_as->setText(QApplication::translate("ClothDesignerClass", "save project as", 0));
        actionSave_as->setShortcut(QApplication::translate("ClothDesignerClass", "Ctrl+Alt+S", 0));
        actionExport_batch_simulation->setText(QApplication::translate("ClothDesignerClass", "export batch simulation", 0));
        actionTraining_image_render->setText(QApplication::translate("ClothDesignerClass", "training image render", 0));
        actionArcsim->setText(QApplication::translate("ClothDesignerClass", "arcsim", 0));
        menuFile->setTitle(QApplication::translate("ClothDesignerClass", "file", 0));
        menuEdit->setTitle(QApplication::translate("ClothDesignerClass", "edit", 0));
        menuHistory->setTitle(QApplication::translate("ClothDesignerClass", "history", 0));
        menuWindows->setTitle(QApplication::translate("ClothDesignerClass", "windows", 0));
        groupBox->setTitle(QApplication::translate("ClothDesignerClass", "Simulation Param", 0));
        label_11->setText(QApplication::translate("ClothDesignerClass", "gravity", 0));
        label_9->setText(QApplication::translate("ClothDesignerClass", "spring stiff", 0));
        cbSelfCollision->setText(QApplication::translate("ClothDesignerClass", "self collision", 0));
        label_10->setText(QApplication::translate("ClothDesignerClass", "bend stiff", 0));
        label_15->setText(QApplication::translate("ClothDesignerClass", "triangle size (mm)", 0));
        groupBox_2->setTitle(QApplication::translate("ClothDesignerClass", "Piece Param", 0));
        label_16->setText(QApplication::translate("ClothDesignerClass", "bend mult", 0));
        label_17->setText(QApplication::translate("ClothDesignerClass", "spring mult", 0));
        label_18->setText(QApplication::translate("ClothDesignerClass", "material", 0));
        groupBox_3->setTitle(QApplication::translate("ClothDesignerClass", "Sew Param", 0));
        label->setText(QApplication::translate("ClothDesignerClass", "angle", 0));
        tabWidget->setTabText(tabWidget->indexOf(tabSimulation), QApplication::translate("ClothDesignerClass", "Simulation", 0));
        pbSaveSmplCoeffs->setText(QApplication::translate("ClothDesignerClass", "save smpl coeffs", 0));
        pbLoadSmplCoeffs->setText(QApplication::translate("ClothDesignerClass", "load smpl coeffs", 0));
        pbResetSmplCoeffs->setText(QApplication::translate("ClothDesignerClass", "reset smpl coeffs", 0));
        gpSmplBodyCoeffs->setTitle(QApplication::translate("ClothDesignerClass", "smpl body coeffs", 0));
        pbBindClothesToSmpl->setText(QApplication::translate("ClothDesignerClass", "bind clothes to smpl", 0));
        pbLoadSmplFromXml->setText(QApplication::translate("ClothDesignerClass", "load smpl from xml", 0));
        tabWidget->setTabText(tabWidget->indexOf(tab), QApplication::translate("ClothDesignerClass", "Body", 0));
        pbFlipPolygon->setText(QApplication::translate("ClothDesignerClass", "flip polygon", 0));
        pbResetSimulation->setText(QApplication::translate("ClothDesignerClass", "reset simulaton", 0));
        pbResetSimulation->setShortcut(QApplication::translate("ClothDesignerClass", "1", 0));
        pbMirrorSelected->setText(QApplication::translate("ClothDesignerClass", "mirror selected", 0));
        pbCopySelected->setText(QApplication::translate("ClothDesignerClass", "copy selected", 0));
    } // retranslateUi

};

namespace Ui {
    class ClothDesignerClass: public Ui_ClothDesignerClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_CLOTHDESIGNER_H
