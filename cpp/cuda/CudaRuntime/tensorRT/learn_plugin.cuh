#pragma once

#include <NvInfer.h>

class MyPlugin : public nvinfer1::IPluginV2Ext
{
	virtual ~MyPlugin() {}
};

class MyPluginCreator : public nvinfer1::IPluginCreator
{

};