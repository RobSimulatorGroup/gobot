/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-3-23
*/

#pragma once

namespace gobot {

class TextureStorage;
class FrameBufferCache;
class RendererSceneRender;
class RendererTextureStorage;


class RendererCompositor {
protected:
    static RendererCompositor*(*CreateFunc)();

public:
    RendererCompositor();

    virtual ~RendererCompositor();

    static RendererCompositor* GetInstance();

    virtual RendererSceneRender* GetScene() = 0;

    virtual RendererTextureStorage* GetTextureStorage() = 0;

    virtual void Initialize() = 0;

    virtual void BeginFrame(double frame_step) = 0;

    virtual void EndFrame(bool p_swap_buffers) = 0;

    virtual void Finalize() = 0;

private:

    static RendererCompositor* s_singleton;

};


}
