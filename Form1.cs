using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;

// 要引用的類別
using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.Util;
using System.Threading;

namespace faceopenCV
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        private Capture _capture;
        private bool _captureInProgress;
        private HaarCascade face;
        private bool isTrack = false;
        private Image<Gray, byte> hue = null;
        private Image<Gray, byte> mask = null;
        private Image<Gray, byte> backproject = null;
        private Image<Hsv, byte> hsv = null;
        private Image<Gray, Byte> gray = null;
        private IntPtr[] img = null;
        private Rectangle trackwin;
        private MCvConnectedComp trackcomp = new MCvConnectedComp();
        private MCvBox2D trackbox = new MCvBox2D();
        
        private DenseHistogram hist = new DenseHistogram(16, new RangeF(0, 180));

        private void ProcessFrame(object sender, EventArgs arg)
        {
            //Image<Bgr, Byte> frame = _capture.QueryFrame();
            //captureImageBox.Image = frame;

            //Image<Gray, Byte> grayFrame = frame.Convert<Gray, Byte>();
            //grayscaleImageBox.Image = grayFrame;
            //Image<Gray, Byte> cannyFrame = frame.Canny(new Gray(100),new Byte(60));
            //cannyImageBox.Image = cannyFrame;
            Image<Bgr, Byte> frame = _capture.QueryFrame();
            if (frame != null) 
            {
                gray = frame.Convert<Gray, Byte>();
                gray._EqualizeHist();
                hue = new Image<Gray, byte>(frame.Width, frame.Height);
                mask = new Image<Gray, byte>(frame.Width, frame.Height);
                backproject = new Image<Gray, byte>(frame.Width, frame.Height);
            }
            hsv = frame.Convert<Hsv, byte>(); //彩色空間轉換從BGR到HSV
            hsv._EqualizeHist();

            HaarCascade face = new HaarCascade("haarcascade_frontalface_alt_tree.xml"); // 戴入haar分類器
            //HaarCascade eye = new HaarCascade("haarcascade_eye.xml");
            Emgu.CV.CvInvoke.cvInRangeS(hsv, new MCvScalar(0, 30, Math.Min(10, 255), 0), new MCvScalar(180, 256, Math.Max(10, 255), 0), mask);
            Emgu.CV.CvInvoke.cvSplit(hsv, hue, IntPtr.Zero, IntPtr.Zero, IntPtr.Zero);
            if (isTrack == false)
            {
                MCvAvgComp[][] facesDetected = gray.DetectHaarCascade(face,1.1,10, 
                    Emgu.CV.CvEnum.HAAR_DETECTION_TYPE.DO_CANNY_PRUNING, new Size(20, 20));
                foreach (MCvAvgComp f in facesDetected[0])
                {
                    frame.Draw(f.rect, new Bgr(Color.Red), 2);
                    Emgu.CV.CvInvoke.cvSetImageROI(hue, f.rect);  // 將選擇區域設置為ROI
                    Emgu.CV.CvInvoke.cvSetImageROI(mask, f.rect);// 將選擇區域設置為ROI
                    img = new IntPtr[1]
                    {
                        hue
                    };
                    Emgu.CV.CvInvoke.cvCalcHist(img, hist, false, mask); //計算直方圖
                    Emgu.CV.CvInvoke.cvResetImageROI(hue); //釋放直方圖
                    Emgu.CV.CvInvoke.cvResetImageROI(mask); //釋放直方圖
                    trackwin = f.rect;
                    isTrack = true;
                }
            }
            img = new IntPtr[1]
            {
                hue
            };
            if (trackwin.Width == 0) trackwin.Width = 40;
            if (trackwin.Height == 0) trackwin.Height = 40;
            Emgu.CV.CvInvoke.cvCalcBackProject(img, backproject, hist); //使用back project方法
            Emgu.CV.CvInvoke.cvAnd(backproject, mask, backproject, IntPtr.Zero);
            Emgu.CV.CvInvoke.cvCamShift(backproject, trackwin, new MCvTermCriteria(10, 0.5), out trackcomp, out trackbox); //使用camshift
            trackwin = trackcomp.rect;

            frame.Draw(trackwin, new Bgr(Color.Red), 2);
            imageBox2.Image = hsv;
            imageBox3.Image = mask;
            imageBox1.Image = frame;
        }

        private void captureButton_Click(object sender, EventArgs e)
        {
            #region if capture is not created, create it now
            if (_capture == null)
            {
                try
                {
                    _capture = new Capture();
                }
                catch (NullReferenceException excpt)
                {
                    MessageBox.Show(excpt.Message);
                }
            }
            #endregion

            if (_capture != null)
            {
                if (_captureInProgress)
                {  //stop the capture
                    Application.Idle -= new EventHandler(ProcessFrame);
                    captureButton.Text = "Start Capture";
                }
                else
                {
                    //start the capture
                    captureButton.Text = "Stop";
                    Application.Idle += new EventHandler(ProcessFrame);
                }

                _captureInProgress = !_captureInProgress;
            }
        }

        private void button1_Click(object sender, EventArgs e)
        {
            Image<Bgr, Byte> frame1 = _capture.QueryFrame();
            imageBox3.Image = frame1;
            //Image<Gray, Byte> grayFrame = frame1.Convert<Gray, Byte>();
            //grayscaleImageBox.Image = grayFrame;
        }

        /*private void Form1_Load(object sender, EventArgs e)
        {
            HaarCascade face = new HaarCascade("haarcascade_frontalface_alt_tree.xml"); // 戴入haar分類器
        }*/
    }
}
