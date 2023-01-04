/* Copyright(c) 2020-2022, Qiqi Wu<1258552199@qq.com>.
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 * This file is created by Qiqi Wu, 22-11-20
*/

#pragma once

#include "gobot/core/types.hpp"

namespace gobot {

class ResourceFormatSaverSceneInstance;

class VariantSerializer {
public:
    static Json VariantToJson(Instance obj, ResourceFormatSaverSceneInstance* resource_format_saver = nullptr);

private:
    static ResourceFormatSaverSceneInstance* s_resource_format_saver_;

    static void ToJsonRecursively(Instance object, Json& writer);

    static bool WriteVariant(const Variant& var, Json& writer);

    static bool WriteAtomicTypesToJson(const Type& t, const Variant& var, Json& writer);

    static void WriteAssociativeContainer(const VariantMapView& view, Json& writer);

    static void WriteArray(const VariantListView& view, Json& writer);
};



}