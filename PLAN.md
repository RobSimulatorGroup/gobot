# Scene Tree Serialization Refactor Plan

## 紧急方向排序

当前优先级按“能最快闭环、能测试、不会被外部 SDK 卡住”排序：

1. **机器人 3D 场景树序列化**：先完成 `PackedScene` / `SceneState` / `.jscn` 的基础闭环，这是 OpenUSD、编辑器保存和物理场景交换的共同底座
2. **OpenUSD 支持**：先做可选依赖和 importer 架构，再逐步落到 `PackedScene`、Mesh、Material、Physics schema
3. **PBR 渲染最小可实现**：在 Mesh/Material 数据模型稳定后接 OpenGL shader 和 material storage
4. **NVIDIA Newton 物理引擎**：外部依赖最重，先抽象物理接口和 USD Physics 映射，再接 SDK

## 现状问题

1. `PackedScene` / `SceneState` 几乎是空壳 — `NodeData` 结构体是空的，`__NODES__` 的 load/save 逻辑都是 stub
2. `ResourceFormatSaverSceneInstance::Save()` 和 `LoadResource()` 混杂了资源序列化和场景序列化两套逻辑，耦合严重
3. `VariantSerializer` 承担了太多职责：类型转换、资源引用解析、属性遍历全混在一起
4. Node 本身没有任何序列化相关的接口，场景树结构（父子关系、节点类型）完全没有被持久化
5. `FindResources()` 里有个 bug：`TypeCategory::Compound` 分支的条件写反了（`!` 应该去掉）

---

## Phase 1 — 定义 SceneState 数据结构

**目标**：让 `SceneState` 能完整描述一棵节点树

```cpp
SceneState::PropertyData {
    string  name
    Variant value
}

SceneState::NodeData {
    string  type        // RTTR 类名，如 "Node3D"
    string  name
    int     parent      // index into nodes_ array, -1 = root
    Ref<PackedScene> instance  // 如果是实例化的子场景
    vector<PropertyData> properties
}
```

新增方法：
- `PackedScene::Pack(Node* root)` — 把一棵活的节点树打包进 SceneState
- `PackedScene::Instantiate()` — 从 SceneState 还原出节点树

---

## Phase 2 — 验证 Node 序列化接口

**目标**：确认 RTTR 反射对 Node 子类的覆盖完整

- 检查各 Node 子类的 `RTTR_REGISTRATION`，确保关键属性都有 `PropertyUsageFlags::Storage` flag
- 确认 `Node::Get` / `Node::Set` 接口对所有需要序列化的属性都能正常工作
- 补齐缺失的注册

---

## Phase 3 — 实现 Pack / Instantiate 核心逻辑

**目标**：完整实现场景树的打包和还原

`Pack(Node* root)` 逻辑：
```
DFS 遍历节点树
  → 记录 type, name, parent_index
  → 遍历 Storage 属性，收集 Variant 值
  → 如果节点是子场景实例，记录 instance 引用
```

`Instantiate()` 逻辑：
```
按 nodes_ 顺序创建节点（RTTR create by type name）
  → 设置属性
  → 按 parent_index 建立父子关系
  → 返回 root node
```

---

## Phase 4 — 重构 resource_format_scene

**目标**：把 `.jscn` 的 save/load 接通 Phase 3，拆分职责

文件格式升级到 version 2：

```json
{
  "__VERSION__": 2,
  "__META_TYPE__": "SCENE",
  "__TYPE__": "PackedScene",
  "__EXT_RESOURCES__": [...],
  "__SUB_RESOURCES__": [...],
  "__NODES__": [
    {
      "type": "Node3D",
      "name": "RobotRoot",
      "parent": -1,
      "properties": { "position": [...] }
    },
    {
      "type": "MeshInstance3D",
      "name": "Body",
      "parent": 0,
      "instance": "ExtResource(1)"
    }
  ]
}
```

拆分职责：
- `ResourceFormatSaverSceneInstance` 只负责 JSON 序列化，不做资源图遍历
- `ResourceFormatLoaderSceneInstance` 只负责 JSON 反序列化
- 资源图遍历逻辑移到独立的 `SceneSerializer` 类

---

## Phase 5 — Editor 接入 + 测试

**目标**：让编辑器的 Save/Load 按钮真正工作，补充测试

- 接通 `src/gobot/editor/imgui/inspector_panel.cpp` 里的 TODO
- 新增 `tests/scene/test_packed_scene.cpp`：Pack → Instantiate 往返测试
- 补充 `tests/core/io/test_resource_format_scene.cpp`：场景文件读写测试

---

## 执行顺序

```
Phase 1 (SceneState 数据结构)
    ↓
Phase 2 (验证 RTTR 覆盖)
    ↓
Phase 3 (Pack / Instantiate 核心逻辑)
    ↓
Phase 4 (文件格式 + IO 重构)
    ↓
Phase 5 (Editor + 测试)
```

每个 Phase 结束后都能编译并跑测试。

---

# OpenUSD Support Plan

## 目标

让 Gobot 可以把 OpenUSD 文件作为机器人/仿真场景交换格式：

- Import：`.usd` / `.usda` / `.usdc` → `PackedScene`
- Export：`PackedScene` → `.usd` / `.usda` / `.usdc`
- 中长期支持：几何、材质、层级变换、实例、物理 schema，并为 NVIDIA Newton 提供场景输入

## 原则

1. OpenUSD 必须是**可选依赖**：未安装 OpenUSD 时，工程仍可编译、测试和运行
2. Import 先于 Export：先解决外部机器人资产进入 Gobot
3. 先支持稳定核心 schema：`UsdGeomXform`、`UsdGeomMesh`、`UsdShadePreviewSurface`
4. 不把 USD 细节泄漏进通用场景接口：USD loader 输出 `PackedScene` / `Resource`

## Phase USD-1 — 可选依赖和 Loader 骨架

**目标**：工程能识别 USD 文件，并在没有 OpenUSD SDK 时给出受控失败

- CMake 新增 `GOB_BUILD_OPENUSD` 开关
- 有 OpenUSD 时定义 `GOBOT_HAS_OPENUSD`
- 新增 `ResourceFormatLoaderUSD`
- 注册扩展名：`usd`、`usda`、`usdc`
- 接入 `SceneInitializer`
- 单元测试覆盖：
  - loader recognized extensions
  - loader handles `PackedScene`
  - 未启用 OpenUSD 时 load 返回空资源，不崩溃

## Phase USD-2 — USD 层级导入

**目标**：把 USD stage 的层级转换为 `PackedScene`

```
UsdStage
  → Traverse prims
  → UsdGeomXformable 转 Node3D
  → 记录 name/type/parent
  → xform ops 转 Node3D transform 属性
```

测试：
- 简单 usda：root + child xform
- 导入后 `PackedScene::Instantiate()` 可还原父子关系

## Phase USD-3 — Mesh Resource

**目标**：新增真正可存储导入网格数据的 Mesh 资源

新增资源建议：

```cpp
ArrayMesh / SurfaceMesh {
    vector<Vector3> positions
    vector<Vector3> normals
    vector<Vector2> uvs
    vector<uint32_t> indices
    Ref<Material> material
}
```

USD 映射：
- `UsdGeomMesh.points` → positions
- `faceVertexIndices` / `faceVertexCounts` → triangulated indices
- `primvars:normals` / `primvars:st` → normals / uvs

## Phase USD-4 — Material Import

**目标**：把 `UsdShadePreviewSurface` 映射到 `PBRMaterial3D`

最小字段：
- `diffuseColor` / `baseColor` → albedo
- `metallic`
- `roughness`
- texture path 先保留引用，后续接 Texture resource

## Phase USD-5 — Export

**目标**：把 Gobot 场景导出为 USD

- `PackedScene` / `SceneState` → `UsdStage`
- `Node3D` transform → `UsdGeomXformOp`
- Mesh resource → `UsdGeomMesh`
- `PBRMaterial3D` → `UsdShadePreviewSurface`

## Phase USD-6 — Physics / Newton Bridge

**目标**：让 USD Physics 成为 Newton 物理场景的输入之一

- `UsdPhysicsRigidBodyAPI` → RigidBody node/resource
- `UsdPhysicsCollisionAPI` → CollisionShape
- mass / inertia / velocity / collision group 映射
- Newton backend 从 Gobot 物理抽象读取，不直接依赖 USD
