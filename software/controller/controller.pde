import g4p_controls.*;
import processing.serial.*;
import java.util.Date;
import java.text.SimpleDateFormat;
import grafica.*;

Serial port;
int BAUD_RATE = 115200;
boolean RECORDING = false;
long RECORDING_START_TIME;
float S_TO_NS = 1e-9;
GPlot PLOT;
GPointsArray PLOT_POINTS;

void setup() {
  size(1200, 800);
  String portName = Serial.list()[0];
  println("Connecting to port "+portName+".");
  port = new Serial(this, portName, BAUD_RATE);
  port.bufferUntil(10); //linefeed
  createGUI();
  connectedLabel.setVisible(false);
  PLOT = new GPlot(this, 100, 100, 1100, 700);
  PLOT_POINTS = new GPointsArray(10000);
  PLOT.setTitleText("Thrust Data");
  PLOT.getXAxis().setAxisLabelText("Time (s)");
  PLOT.getYAxis().setAxisLabelText("Thurst (N)");
  PLOT.setPoints(PLOT_POINTS);
}

void draw() {
  clear();
  background(200);
  PLOT.setPoints(PLOT_POINTS);
  PLOT.defaultDraw();
}

void stop() {
  port.stop();
}

void serialEvent(Serial p) { 
  String s = p.readString();
  s = s.substring(0, s.length()-2);
  if (RECORDING) {
    PLOT_POINTS.add((java.lang.System.nanoTime()-RECORDING_START_TIME)*S_TO_NS, float(s));
  } else {
    if (s.equals("I/O RESPONSE FROM HARDWARE")) {
      disconnectedLabel.setVisible(false);
      connectedLabel.setVisible(true);
    }
  }
} 
