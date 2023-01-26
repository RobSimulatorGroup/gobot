/* Copyright(c) 2020-2022, Qiqi Wu<1258552199@qq.com>.
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 * This file is created by Qiqi Wu, 22-11-20
*/

#pragma once

#include "gobot/core/types.hpp"
#include "gobot/core/marcos.hpp"

namespace gobot {

class ResourceFormatSaverSceneInstance;
class ResourceFormatLoaderSceneInstance;

class GOBOT_API VariantSerializer {
public:
    static Json VariantToJson(const Variant& variant,
                              ResourceFormatSaverSceneInstance* resource_format_saver = nullptr);

    static Variant JsonToVariant(const Type& type,
                                 const Json& json,
                                 ResourceFormatLoaderSceneInstance* s_resource_format_loader = nullptr);

private:

    // For Save
    static ResourceFormatSaverSceneInstance* s_resource_format_saver_;

    // For Load
    static ResourceFormatLoaderSceneInstance* s_resource_format_loader_;

    //////////////////////////////////////////
    static void ToJsonRecursively(Instance object, Json& writer);

    // return is continue or not
    static bool SaveResource(Instance instance, const Type& t, Json& writer);

    static bool WriteVariant(const Variant& var, Json& writer);

    static bool WriteAtomicTypesToJson(const Type& t, const Variant& var, Json& writer);

    static void WriteAssociativeContainer(const VariantMapView& view, Json& writer);

    static void WriteArray(const VariantListView& view, Json& writer);

    ///////////////////////////////////////////

    static void LoadSubResource(Instance obj, const String& uuid);

    static void LoadExtResource(Instance obj, const String& uuid);

    static void FromJsonRecursively(Instance instance, const Json& json);

    static Variant ExtractPrimitiveTypes(const Type& type, const Json& json_value);

    static void WriteArrayRecursively(VariantListView& view, const Json& json_array_value);

    static void WriteAssociativeViewRecursively(VariantMapView& view, const Json& json_array_value);

    static Variant ExtractValue(const Type& type, const Json& json);

    static void LoadResource(Instance obj, const Type& t, const Json& json);

};



}