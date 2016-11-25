// SetConfigAndRun.cpp : 定義主控台應用程式的進入點。
//

#include "stdafx.h"
#include <iostream>
#include "stdlib.h"
#include "stdio.h"
#include <windows.h>
#include <fstream>
#include <string>
using namespace std;
#define QuantityOfConfig 15

string Split(string &str, string &STARTDELIMITER, string &STOPDELIMITER)
{
	//string str = "STARTDELIMITER_0_192.168.1.18_STOPDELIMITER";
	unsigned first = str.find(STARTDELIMITER);
	unsigned last = str.find(STOPDELIMITER);
	string strNew = str.substr(first + STARTDELIMITER.length(), last - (first + STARTDELIMITER.length()));
	str.erase(0,last + STOPDELIMITER.length() - first);
	return strNew;
}


int _tmain(int argc, _TCHAR* argv[])
{
	string file_contents;
	ifstream file("RunConfig.txt");
	string str;
	int readfile_count = 0;
	string start_del, end_del = "\n", result[QuantityOfConfig];
	std::string content((std::istreambuf_iterator<char>(file)),
	(std::istreambuf_iterator<char>()));
	start_del = "Index=";
	result[0] = Split(content, start_del, end_del);
	start_del = "QuantityOfFile=";
	result[1] = Split(content, start_del, end_del);
	start_del = "--------------BoundingBox---------------";
	Split(content, start_del, end_del);
	start_del = "BountdingBox_Erode_iter=";
	result[2] = Split(content, start_del, end_del);
	start_del = "BountdingBox_Dilate_iter=";
	result[3] = Split(content, start_del, end_del);
	start_del = "image_100x100_Dilate=";
	result[4] = Split(content, start_del, end_del);
	start_del = "image_100x100_Erode=";
	result[5] = Split(content, start_del, end_del);
	start_del = "-------------Preprocessing--------------";
	Split(content, start_del, end_del);
	start_del = "Rotateenable=";
	result[6] = Split(content, start_del, end_del);
	start_del = "Scalingenable=";
	result[7] = Split(content, start_del, end_del);
	start_del = "Shiftenable=";
	result[8] = Split(content, start_del, end_del);
	start_del = "Crop16x16enable=";
	result[9] = Split(content, start_del, end_del);
	start_del = "layer_num=";
	result[10] = Split(content, start_del, end_del);
	start_del = "rotate_angle=";
	result[11] = Split(content, start_del, end_del);
	start_del = "x_offset_val=";
	result[12] = Split(content, start_del, end_del);
	start_del = "y_offset_val=";
	result[13] = Split(content, start_del, end_del);
	start_del = "----------------------------------------";
	Split(content, start_del, end_del);
	start_del = "Index_end=";
	result[14] = Split(content, start_del, end_del);

	int initial_num = stoi(result[0]);
	int end_num = stoi(result[14]);
	for (int i = initial_num; i <= end_num; i++)
	{
		if (i != 6 && i != 7 && i != 9)
		{
			string Output_Dir_Tag = result[1] + "_" + result[2] + "_"
				+ result[3] + "_" + result[4] + "_" + result[5] + "_"
				+ result[6] + "_" + result[7] + "_" + result[8] + "_"
				+ result[9] + "_" + result[10] + "_" + result[11] + "_"
				+ result[12] + "_" + result[13];
			string index_int = to_string(i);
			ofstream BoundingBoxfile("D:\\Skywawtch\\Working\\caffe\\prediction\\BoundingBoxParam.txt");
			if (BoundingBoxfile.is_open())
			{
				BoundingBoxfile << index_int + "\n";
				BoundingBoxfile << result[1] + "\n";
				BoundingBoxfile << result[2] + "\n";
				BoundingBoxfile << result[3] + "\n";
				BoundingBoxfile << result[4] + "\n";
				BoundingBoxfile << result[5];
				BoundingBoxfile.close();
			}
			else cout << "Unable to open file";

			ofstream Shiftfile("D:\\Skywawtch\\Working\\caffe\\prediction\\ShiftParam.txt");
			if (Shiftfile.is_open())
			{
				Shiftfile << index_int + "\n";
				Shiftfile << result[1] + "\n";
				Shiftfile << result[6] + "\n";
				Shiftfile << result[7] + "\n";
				Shiftfile << result[8] + "\n";
				Shiftfile << result[9] + "\n";
				Shiftfile << result[10] + "\n";
				Shiftfile << result[11] + "\n";
				Shiftfile << result[12] + "\n";
				Shiftfile << result[13];
				Shiftfile.close();
			}
			ofstream ClassConfigfile("D:\\Skywawtch\\Working\\caffe\\python\\classify_config.txt");
			if (ClassConfigfile.is_open())
			{
				ClassConfigfile << result[1] + "\n";
				int QuantityOfSubfile = 0;
				if (result[6].compare("1") == 0) // Rotateenable
				{
					QuantityOfSubfile++;
				}
				if (result[7].compare("1") == 0) // Scalingenable
				{
					QuantityOfSubfile++;
				}
				if (result[8].compare("1") == 0) // Shiftenable
				{
					int temp = stoi(result[10]);
					temp = temp * 4;
					QuantityOfSubfile += temp;
				}
				if (result[8].compare("1") == 0) // Crop16x16enable
				{
					QuantityOfSubfile++;
				}
				string output = to_string(QuantityOfSubfile);
				ClassConfigfile << output + "\n";
				ClassConfigfile << index_int + "\n";
				ClassConfigfile << Output_Dir_Tag;

				ClassConfigfile.close();
			}


			else cout << "Unable to open file";
			char SystemCommand[1024] = { 0 };

			sprintf(SystemCommand, "D:\\Skywawtch\\Working\\caffe\\prediction\\GetBoundingBox.exe");
			system(SystemCommand);
			sprintf(SystemCommand, "D:\\Skywawtch\\Working\\caffe\\prediction\\ShiftImage.exe");
			system(SystemCommand);
			sprintf(SystemCommand, "python D:\\Skywawtch\\Working\\caffe\\python\\classify_ok_multi_multi.py --model_def D:\\Skywawtch\\Working\\caffe\\python\\lenet.prototxt --pretrained_model D:\\Skywawtch\\Working\\caffe\\python\\lenet_iter_80000LMDB.caffemodel --force_grayscale --center_only --print_results D:\\Skywawtch\\Working\\caffe\\%d_BoundingBox\\ImageSet\\ outyoyo", i);
			system(SystemCommand);
			//printf(SystemCommand);
			//------------------------------------------analysis-------------------------------------------
			string file_contents;
			char read_analysis_path[1024];
			sprintf(read_analysis_path,"D:\\Skywawtch\\Working\\caffe\\prediction\\prediction_file%s\\prediction_file%d.txt", Output_Dir_Tag.c_str(), i);
			ifstream FileAnalysis(read_analysis_path);
			string str_analysis;
			int readfile_count = 0;
			int hit_count = 0;
			while (std::getline(FileAnalysis, str_analysis))
			{
				readfile_count++;
				int analysis_integer = stoi(str_analysis);
				if (analysis_integer == i)
				{
					hit_count++;
				}
			}
			float Percentage = (float)((float)hit_count / (float)readfile_count);
			string output_percentage_string = to_string(Percentage);
			char output_analysis_path[1024];
			sprintf(output_analysis_path, "D:\\Skywawtch\\Working\\caffe\\prediction\\prediction_file%s\\file%d_percentage.txt", Output_Dir_Tag.c_str(), i);
			ofstream PercentageFile(output_analysis_path);
			if (PercentageFile.is_open())
			{
				PercentageFile << output_percentage_string;
				PercentageFile.close();
			}
			
		}
	}
	return 0;
}

