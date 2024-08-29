#pragma once

#include <NvInfer.h>
#include <iostream>

class Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity <= Severity::kVERBOSE)
            std::cout << msg << std::endl;
    }

    virtual ~Logger() {}
};

void learn_rt();

