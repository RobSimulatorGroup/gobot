/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-20
*/

#pragma once

#include <cstdint>

namespace gobot {

class RenderPass;
class Framebuffer;
class Pipeline;

class CommandBuffer
{
public:
    virtual ~CommandBuffer() = default;

    static CommandBuffer* Create();

    virtual bool Init(bool primary = true)                                                  = 0;
    virtual void Unload()                                                                   = 0;
    virtual void BeginRecording()                                                           = 0;
    virtual void BeginRecordingSecondary(RenderPass* renderPass, Framebuffer* framebuffer)  = 0;
    virtual void EndRecording()                                                             = 0;
    virtual void ExecuteSecondary(CommandBuffer* primaryCmdBuffer)                          = 0;
    virtual void UpdateViewport(uint32_t width, uint32_t height, bool flipViewport = false) = 0;
    virtual bool Flush() { return true; }
    virtual void Submit() { }
    virtual void BindPipeline(Pipeline* pipeline) = 0;
    virtual void UnBindPipeline()                 = 0;

protected:
    static CommandBuffer* (*CreateFunc)();
};

}