/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-6-15
*/

#pragma once

#include "gobot/core/ref_counted.hpp"
#include "gobot/core/io/image.hpp"
#include "gobot/core/io/resource_loader.hpp"

namespace gobot {

class ImageLoader;

enum class LoaderFlags {
    None = 0,
    ForceLinear = 1,
    ConvertColors = 2,
};

// Because image can have different types(.jpg, .png, etc.), we can't use ResourceFormatLoader directly.
// The ImageFormatLoader is the a class inherits from RefCounted just like ResourceFormatLoader, then we
// others can inherit from ImageFormatLoader for different image types.
class ImageFormatLoader : public RefCounted {
    GOBCLASS(ImageFormatLoader, RefCounted);
    friend class ImageLoader;
    friend class ResourceFormatLoaderImage;
protected:
    virtual Ref<Image> LoadImage(const ByteArray &byte_array,
                                 LoaderFlags flags = LoaderFlags::None,
                                 float scale = 1.0) = 0;

    virtual void GetRecognizedExtensions(std::vector<String>* extensions) const = 0;

    bool Recognize(const String &extension) const;

public:
    virtual ~ImageFormatLoader() {}
};


class ImageLoader {
    static std::vector<Ref<ImageFormatLoader>> s_loaders;
    friend class ResourceFormatLoaderImage;

protected:
public:
    static Ref<Image> LoadImage(const String& path,
                                LoaderFlags flags = LoaderFlags::None,
                                float scale = 1.0);

    static void GetRecognizedExtensions(std::vector<String>* extensions);

    static Ref<ImageFormatLoader> Recognize(const String& extension);

    static void AddImageFormatLoader(Ref<ImageFormatLoader> loader);

    static void RemoveImageFormatLoader(Ref<ImageFormatLoader> loader);

    static void Cleanup();
};

}