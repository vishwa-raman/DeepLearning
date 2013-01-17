#include <QApplication>
#include <QFont>
#include <QLCDNumber>
#include <QPushButton>
#include <QSlider>
#include <QVBoxLayout>
#include <QWidget>

class MyWidget : public QWidget
{
public:
  MyWidget(QWidget *parent = 0);
};

MyWidget::MyWidget(QWidget *parent)
  : QWidget(parent) {
  QPushButton *quit = new QPushButton(tr("Quit"));
  quit->setFont(QFont("Times", 18, QFont::Bold));

  QLCDNumber *lcd = new QLCDNumber(2);
  lcd->setSegmentStyle(QLCDNumber::Filled);

  QSlider *slider = new QSlider(Qt::Horizontal);
  slider->setRange(0, 99);
  slider->setValue(0);

  connect(quit, SIGNAL(clicked()), qApp, SLOT(quit()));
  connect(slider, SIGNAL(valueChanged(int)),
	  lcd, SLOT(display(int)));

  QVBoxLayout *layout = new QVBoxLayout;
  layout->addWidget(quit);
  layout->addWidget(lcd);
  layout->addWidget(slider);
  setLayout(layout);
}

int main(int argc, char *argv[]) {
  QApplication app(argc, argv);
  MyWidget widget;
  widget.show();
  return app.exec();
}
