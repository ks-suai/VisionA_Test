#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

int main() {
    string imagePath = "D:/VisionA_Test/led_digit.jpg"; 
    Mat img = imread(imagePath);
    if (img.empty()) img = imread("led_digit.jpg");
    if (img.empty()) return -1;

    Mat gray, binary;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    // 把阈值拉高，尽量减少光晕干扰
    threshold(gray, binary, 220, 255, THRESH_BINARY); 

    // 找边界
    int top = binary.rows, bottom = 0, left = binary.cols, right = 0;
    for (int i = 0; i < binary.rows; i++) {
        if (countNonZero(binary.row(i)) > 5) { top = min(top, i); bottom = max(bottom, i); }
    }
    for (int j = 0; j < binary.cols; j++) {
        if (countNonZero(binary.col(j)) > 5) { left = min(left, j); right = max(right, j); }
    }

    Mat resultImg = img.clone();

    if (right > left && bottom > top) {
        Rect rect(left, top, right - left, bottom - top);
        rectangle(resultImg, rect, Scalar(0, 255, 0), 3);

        Mat roi = binary(rect);
        
        // 降维到 32x32 (黄金尺寸)
        Mat small_roi;
        resize(roi, small_roi, Size(32, 32), 0, 0, INTER_AREA);

        // ================== 核心：统计每一行是否有横排 ==================
        int active_rows = 0;
        cout << "--- 32x32 Grid ---" << endl;
        for (int i = 0; i < 32; i++) {
            int row_pixels = countNonZero(small_roi.row(i));
            // 如果这一行的白点数量 > 8 个，我们认为这是一条横线
            if (row_pixels > 8) {
                active_rows++;
                cout << "Row " << i << ": [========] (横线)" << endl;
            } else if (row_pixels > 2) {
                cout << "Row " << i << ": [  ..  ]" << endl; // 零星点
            } else {
                cout << "Row " << i << ": [      ]" << endl; // 空行
            }
        }

        cout << "Total Horizontal Rows: " << active_rows << endl;

        // ================== 智能分类逻辑 ==================
        string digitStr = "Unknown";
        
        // 根据你这三张图的特征：
        // 数字1：竖排，横线极少 (通常 <= 6)
        // 数字2：顶部横线，底部横线 (通常 7~10)
        // 数字3：中间断了或者横排太多 (通常 >= 11)
        if (active_rows <= 6) {
            digitStr = "1";
        } else if (active_rows <= 10) {
            digitStr = "2";
        } else {
            digitStr = "3";
        }

        cout << "-> Prediction: " << digitStr << endl << endl;
        putText(resultImg, "Digit: " + digitStr, Point(rect.x, rect.y - 10), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255), 2);
    } else {
        cout << "No valid digit detected." << endl;
    }

    namedWindow("Original", WINDOW_NORMAL);
    namedWindow("Result", WINDOW_NORMAL);
    imshow("Original", img);
    imshow("Result", resultImg);

    waitKey(0);
    destroyAllWindows();
    return 0;
}