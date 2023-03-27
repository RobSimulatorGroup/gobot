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
class SceneRenderer;


class RendererCompositor {
public:
    RendererCompositor();

    virtual ~RendererCompositor();

    static RendererCompositor* GetInstance();

    TextureStorage* GetTextureStorage();

    SceneRenderer* GetSceneRenderer();

private:
    TextureStorage* texture_storage_ = nullptr;
    FrameBufferCache* frame_buffer_cache_ = nullptr;
    SceneRenderer* scene_ = nullptr;

    static RendererCompositor* s_singleton;

};


}
