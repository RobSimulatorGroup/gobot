#include "manual_bindings_internal.hpp"

namespace gobot::python {

std::uint64_t ActiveSceneEpoch() {
    const std::uint64_t scene_script_epoch = PythonScriptRunner::GetExecutingSceneScriptEpoch();
    if (scene_script_epoch != 0) {
        return scene_script_epoch;
    }
    return GetActiveAppContext().GetSceneEpoch();
}

EngineContext* ActiveSceneContext() {
    if (PythonScriptRunner::GetExecutingSceneScriptEpoch() != 0) {
        return GetActiveAppContextOrNull();
    }
    return &GetActiveAppContext();
}

Node* ActiveSceneRoot() {
    if (Node* scene_script_root = PythonScriptRunner::GetExecutingSceneScriptRoot()) {
        return scene_script_root;
    }
    return GetActiveAppContext().GetSceneRoot();
}

std::uint64_t SceneEpochForContext(EngineContext* context) {
    const std::uint64_t scene_script_epoch = PythonScriptRunner::GetExecutingSceneScriptEpoch();
    if (scene_script_epoch != 0 && context == GetActiveAppContextOrNull()) {
        return scene_script_epoch;
    }
    if (context != nullptr) {
        return context->GetSceneEpoch();
    }
    return ActiveSceneEpoch();
}

Node* SceneRootForContext(EngineContext& context) {
    if (PythonScriptRunner::GetExecutingSceneScriptEpoch() != 0 &&
        &context == GetActiveAppContextOrNull()) {
        if (Node* scene_script_root = PythonScriptRunner::GetExecutingSceneScriptRoot()) {
            return scene_script_root;
        }
    }
    return context.GetSceneRoot();
}

bool IsSceneScriptRuntimeMutation() {
    return PythonScriptRunner::IsExecutingSceneScript();
}

[[noreturn]] void ThrowReferenceError(const std::string& message) {
    PyErr_SetString(PyExc_ReferenceError, message.c_str());
    throw py::error_already_set();
}

class ScopedActiveAppContext {
public:
    explicit ScopedActiveAppContext(EngineContext* context) {
        previous_context_ = GetActiveAppContextOrNull();
        if (context != nullptr && context != previous_context_) {
            SetActiveAppContext(context);
            changed_ = true;
        }
    }

    ~ScopedActiveAppContext() {
        if (changed_) {
            SetActiveAppContext(previous_context_);
        }
    }

private:
    EngineContext* previous_context_{nullptr};
    bool changed_{false};
};

Robot3D* CreateTestRobotScene() {
    auto* robot = Object::New<Robot3D>();
    robot->SetName("robot");

    auto* base_link = Object::New<Link3D>();
    base_link->SetName("base");
    base_link->SetPosition({0.0, 0.0, 1.0});
    base_link->SetHasInertial(true);
    base_link->SetMass(1.0);

    auto* collision_shape = Object::New<CollisionShape3D>();
    collision_shape->SetName("base_collision");
    auto box = MakeRef<BoxShape3D>();
    box->SetSize({0.5, 0.5, 0.5});
    collision_shape->SetShape(box);
    base_link->AddChild(collision_shape);

    auto* joint = Object::New<Joint3D>();
    joint->SetName("joint");
    joint->SetJointType(JointType::Revolute);
    joint->SetParentLink("base");
    joint->SetChildLink("tip");
    joint->SetLowerLimit(-1.0);
    joint->SetUpperLimit(1.0);

    auto* tip_link = Object::New<Link3D>();
    tip_link->SetName("tip");

    robot->AddChild(base_link);
    robot->AddChild(joint);
    joint->AddChild(tip_link);
    return robot;
}

CollisionShape3D* CreateBoxCollision(const std::string& name,
                                      const Vector3& size,
                                      const Vector3& position) {
    auto* collision_shape = Object::New<CollisionShape3D>();
    collision_shape->SetName(name);
    collision_shape->SetPosition(position);
    auto box = MakeRef<BoxShape3D>();
    box->SetSize(size);
    collision_shape->SetShape(box);
    return collision_shape;
}

MeshInstance3D* CreateBoxVisual(const std::string& name,
                                const Vector3& size,
                                const Vector3& position) {
    auto* mesh_instance = Object::New<MeshInstance3D>();
    mesh_instance->SetName(name);
    mesh_instance->SetPosition(position);
    auto box = MakeRef<BoxMesh>();
    box->SetSize(size);
    mesh_instance->SetMesh(box);
    return mesh_instance;
}

Node* LoadSceneRoot(const std::string& scene_path) {
    Ref<Resource> resource =
            ResourceLoader::Load(scene_path, "PackedScene", ResourceFormatLoader::CacheMode::Ignore);
    Ref<PackedScene> packed_scene = dynamic_pointer_cast<PackedScene>(resource);
    if (!packed_scene.IsValid()) {
        throw std::runtime_error("failed to load PackedScene from '" + scene_path + "'");
    }

    Node* scene_root = packed_scene->Instantiate();
    if (scene_root == nullptr) {
        throw std::runtime_error("failed to instantiate PackedScene from '" + scene_path + "'");
    }

    return scene_root;
}

bool SaveSceneRoot(Node* root, const std::string& path) {
    if (root == nullptr) {
        throw std::invalid_argument("cannot save a null Gobot scene root");
    }
    Ref<PackedScene> packed_scene = MakeRef<PackedScene>();
    if (!packed_scene->Pack(root)) {
        return false;
    }
    USING_ENUM_BITWISE_OPERATORS;
    const std::string global_path = ProjectSettings::GetInstance()->GlobalizePath(path);
    return ResourceSaver::Save(packed_scene,
                               global_path,
                               ResourceSaverFlags::ReplaceSubResourcePaths |
                                       ResourceSaverFlags::ChangePath);
}

std::string LocalizeResourcePath(const std::string& path) {
    return ProjectSettings::GetInstance()->LocalizePath(path);
}

std::string GlobalizeResourcePath(const std::string& path) {
    return ProjectSettings::GetInstance()->GlobalizePath(LocalizeResourcePath(path));
}

std::string ResourcePathStem(const std::string& path) {
    return std::filesystem::path(GlobalizeResourcePath(path)).stem().string();
}

std::optional<std::string> ReadFirstMJCFIncludeFile(const std::string& xml_path) {
    std::ifstream input(GlobalizeResourcePath(xml_path));
    if (!input.is_open()) {
        return std::nullopt;
    }

    const std::string xml((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    static const std::regex include_regex(R"gobot(<\s*include\b([^>]*)>)gobot", std::regex::icase);
    static const std::regex file_regex(R"gobot(file\s*=\s*["']([^"']+)["'])gobot", std::regex::icase);

    std::smatch include_match;
    if (!std::regex_search(xml, include_match, include_regex)) {
        return std::nullopt;
    }

    std::smatch file_match;
    const std::string attributes = include_match.size() > 1 ? include_match[1].str() : std::string();
    if (!std::regex_search(attributes, file_match, file_regex) || file_match.size() < 2) {
        return std::nullopt;
    }

    return file_match[1].str();
}

std::string GetXmlAttribute(const std::string& attributes, const std::string& name) {
    const std::regex attr_regex("(^|[[:space:]])" + name + R"(\s*=\s*["']([^"']*)["'])", std::regex::icase);
    std::smatch match;
    if (!std::regex_search(attributes, match, attr_regex) || match.size() < 3) {
        return {};
    }
    return match[2].str();
}

std::vector<RealType> ParseRealList(const std::string& text) {
    std::vector<RealType> values;
    std::istringstream input(text);
    RealType value{};
    while (input >> value) {
        values.push_back(value);
    }
    return values;
}

Color ParseColorOrDefault(const std::string& text, const Color& fallback) {
    const std::vector<RealType> values = ParseRealList(text);
    if (values.size() < 3) {
        return fallback;
    }
    return Color(static_cast<float>(values[0]),
                 static_cast<float>(values[1]),
                 static_cast<float>(values[2]),
                 values.size() >= 4 ? static_cast<float>(values[3]) : 1.0f);
}

std::vector<std::pair<std::string, std::string>> ReadMJCFPlaneGeoms(const std::string& xml_path) {
    std::ifstream input(GlobalizeResourcePath(xml_path));
    if (!input.is_open()) {
        return {};
    }

    const std::string xml((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    static const std::regex geom_regex(R"gobot(<\s*geom\b([^>]*)>)gobot", std::regex::icase);

    std::vector<std::pair<std::string, std::string>> plane_geoms;
    auto begin = std::sregex_iterator(xml.begin(), xml.end(), geom_regex);
    auto end = std::sregex_iterator();
    for (auto iter = begin; iter != end; ++iter) {
        const std::string attributes = iter->size() > 1 ? (*iter)[1].str() : std::string();
        if (ToLower(GetXmlAttribute(attributes, "type")) != "plane") {
            continue;
        }

        std::string name = GetXmlAttribute(attributes, "name");
        if (name.empty()) {
            name = "ground";
        }
        plane_geoms.emplace_back(std::move(name), attributes);
    }
    return plane_geoms;
}

std::optional<std::string> ResolveFirstMJCFIncludeResourcePath(const std::string& xml_path) {
    std::optional<std::string> include_file = ReadFirstMJCFIncludeFile(xml_path);
    if (!include_file.has_value() || include_file->empty()) {
        return std::nullopt;
    }

    if (include_file->find("://") != std::string::npos || IsAbsolutePath(*include_file)) {
        return LocalizeResourcePath(*include_file);
    }

    const std::string local_xml_path = LocalizeResourcePath(xml_path);
    return SimplifyPath(PathJoin(GetBaseDir(local_xml_path), *include_file));
}

std::string MakeIncludedScenePath(const std::string& scene_path, const std::string& included_xml_path) {
    const std::string local_scene_path = LocalizeResourcePath(scene_path);
    return SimplifyPath(PathJoin(GetBaseDir(local_scene_path), ResourcePathStem(included_xml_path) + ".jscn"));
}

void AddMJCFScenePlaneGeoms(Node3D* scene_root, const std::string& xml_path) {
    if (scene_root == nullptr) {
        return;
    }

    for (const auto& [name, attributes] : ReadMJCFPlaneGeoms(xml_path)) {
        const std::vector<RealType> size_values = ParseRealList(GetXmlAttribute(attributes, "size"));
        const RealType size_x = size_values.size() >= 1 ? size_values[0] : 50.0;
        const RealType size_y = size_values.size() >= 2 ? size_values[1] : size_x;
        constexpr RealType kGroundThickness = 0.02;
        const Vector3 ground_size{
                std::max<RealType>(size_x * 2.0, 0.01),
                std::max<RealType>(size_y * 2.0, 0.01),
                kGroundThickness};
        const std::vector<RealType> pos_values = ParseRealList(GetXmlAttribute(attributes, "pos"));
        Vector3 position{0.0, 0.0, -kGroundThickness * 0.5};
        if (pos_values.size() >= 3) {
            position = Vector3{pos_values[0],
                               pos_values[1],
                               static_cast<RealType>(pos_values[2] - kGroundThickness * 0.5)};
        }

        const Color color = ParseColorOrDefault(GetXmlAttribute(attributes, "rgba"), Color(0.45f, 0.45f, 0.45f, 1.0f));

        auto* ground = Object::New<Node3D>();
        ground->SetName(name);
        ground->SetPosition(position);

        auto* visual = Object::New<MeshInstance3D>();
        visual->SetName(name + "_visual");
        visual->SetPosition(Vector3{0.0, 0.0, kGroundThickness * 0.5});
        auto mesh = MakeRef<PlaneMesh>();
        mesh->SetSize(Vector2{ground_size.x(), ground_size.y()});
        auto material = MakeRef<PBRMaterial3D>();
        material->SetAlbedo(color);
        mesh->SetMaterial(dynamic_pointer_cast<Material>(material));
        visual->SetMesh(mesh);
        visual->SetSurfaceColor(color);
        ground->AddChild(visual);

        auto* collision = Object::New<CollisionShape3D>();
        collision->SetName(name + "_collision");
        auto shape = MakeRef<BoxShape3D>();
        shape->SetSize(ground_size);
        collision->SetShape(shape);
        ground->AddChild(collision);

        scene_root->AddChild(ground);
    }
}

Ref<PythonScript> LoadPythonScriptResource(const std::string& script_path) {
    Ref<PythonScript> script = dynamic_pointer_cast<PythonScript>(
            ResourceLoader::Load(script_path, "PythonScript", ResourceFormatLoader::CacheMode::Reuse));
    if (!script.IsValid()) {
        throw std::runtime_error("failed to load Python script '" + script_path + "'");
    }
    return script;
}

void SetOptionalNodeScript(Node* root, const std::optional<std::string>& script_path) {
    if (script_path.has_value() && !script_path->empty()) {
        root->SetScript(LoadPythonScriptResource(*script_path));
    }
}

void CopyMJCFDynamicProperties(Node* target, const Node* source) {
    if (target == nullptr || source == nullptr) {
        return;
    }

    if (auto* target_joint = Object::PointerCastTo<Joint3D>(target)) {
        const auto* source_joint = Object::PointerCastTo<Joint3D>(source);
        if (source_joint != nullptr) {
            target_joint->SetJointPosition(source_joint->GetJointPosition());
            target_joint->SetInitialPosition(source_joint->GetInitialPosition());
            target_joint->SetDamping(source_joint->GetDamping());
            target_joint->SetDriveMode(source_joint->GetDriveMode());
            target_joint->SetDriveStiffness(source_joint->GetDriveStiffness());
            target_joint->SetDriveDamping(source_joint->GetDriveDamping());
            target_joint->SetControlLowerLimit(source_joint->GetControlLowerLimit());
            target_joint->SetControlUpperLimit(source_joint->GetControlUpperLimit());
            target_joint->SetForceLowerLimit(source_joint->GetForceLowerLimit());
            target_joint->SetForceUpperLimit(source_joint->GetForceUpperLimit());
            target_joint->SetGear(source_joint->GetGear());
        }
    }

    if (auto* target_collision = Object::PointerCastTo<CollisionShape3D>(target)) {
        const auto* source_collision = Object::PointerCastTo<CollisionShape3D>(source);
        if (source_collision != nullptr) {
            target_collision->SetFriction(source_collision->GetFriction());
            target_collision->SetContactType(source_collision->GetContactType());
            target_collision->SetContactAffinity(source_collision->GetContactAffinity());
            target_collision->SetContactDimension(source_collision->GetContactDimension());
            target_collision->SetSolref(source_collision->GetSolref());
            target_collision->SetSolimp(source_collision->GetSolimp());
            target_collision->SetMargin(source_collision->GetMargin());
            target_collision->SetGap(source_collision->GetGap());
        }
    }

    std::unordered_map<std::string, const Node*> source_children_by_name;
    for (std::size_t i = 0; i < source->GetChildCount(); ++i) {
        const Node* source_child = source->GetChild(static_cast<int>(i));
        source_children_by_name[source_child->GetName()] = source_child;
    }

    for (std::size_t i = 0; i < target->GetChildCount(); ++i) {
        Node* target_child = target->GetChild(static_cast<int>(i));
        auto source_iter = source_children_by_name.find(target_child->GetName());
        if (source_iter != source_children_by_name.end()) {
            CopyMJCFDynamicProperties(target_child, source_iter->second);
        }
    }
}

Node* InstantiateMJCFRoot(const std::string& xml_path) {
    Ref<Resource> resource =
            ResourceLoader::Load(xml_path, "PackedScene", ResourceFormatLoader::CacheMode::Ignore);
    Ref<PackedScene> packed_scene = dynamic_pointer_cast<PackedScene>(resource);
    if (!packed_scene.IsValid()) {
        throw std::runtime_error("failed to import MJCF scene from '" + xml_path + "'");
    }

    Node* root = packed_scene->Instantiate();
    if (root == nullptr) {
        throw std::runtime_error("failed to instantiate MJCF scene from '" + xml_path + "'");
    }

    if (auto* robot = Object::PointerCastTo<Robot3D>(root)) {
        robot->SetSourcePath(LocalizeResourcePath(xml_path));
    }

    return root;
}

void SaveSingleMJCFScene(const std::string& xml_path,
                         const std::string& scene_path,
                         const std::optional<std::string>& name,
                         const std::optional<std::string>& script_path) {
    Node* root = InstantiateMJCFRoot(xml_path);

    try {
        if (name.has_value() && !name->empty()) {
            root->SetName(*name);
        }

        SetOptionalNodeScript(root, script_path);

        if (!SaveSceneRoot(root, scene_path)) {
            throw std::runtime_error("failed to save Gobot scene to '" + scene_path + "'");
        }
    } catch (...) {
        Object::Delete(root);
        throw;
    }

    Object::Delete(root);
}

bool TrySaveSplitMJCFScene(const std::string& xml_path,
                           const std::string& scene_path,
                           const std::optional<std::string>& name,
                           const std::optional<std::string>& script_path) {
    std::optional<std::string> included_xml_path = ResolveFirstMJCFIncludeResourcePath(xml_path);
    if (!included_xml_path.has_value()) {
        return false;
    }

    const std::string robot_scene_path = MakeIncludedScenePath(scene_path, *included_xml_path);
    if (LocalizeResourcePath(robot_scene_path) == LocalizeResourcePath(scene_path)) {
        return false;
    }

    Node* robot_root = InstantiateMJCFRoot(*included_xml_path);
    Node* compiled_scene_root = InstantiateMJCFRoot(xml_path);
    std::string robot_name = robot_root->GetName().empty() ? ResourcePathStem(*included_xml_path) : robot_root->GetName();

    try {
        CopyMJCFDynamicProperties(robot_root, compiled_scene_root);
        if (robot_root->GetName().empty()) {
            robot_root->SetName(robot_name);
        }
        if (!SaveSceneRoot(robot_root, robot_scene_path)) {
            throw std::runtime_error("failed to save Gobot robot asset to '" + robot_scene_path + "'");
        }
    } catch (...) {
        Object::Delete(robot_root);
        Object::Delete(compiled_scene_root);
        throw;
    }
    Object::Delete(robot_root);
    Object::Delete(compiled_scene_root);

    Ref<PackedScene> robot_scene = dynamic_pointer_cast<PackedScene>(
            ResourceLoader::Load(robot_scene_path, "PackedScene", ResourceFormatLoader::CacheMode::Replace));
    if (!robot_scene.IsValid()) {
        throw std::runtime_error("failed to load generated Gobot robot asset '" + robot_scene_path + "'");
    }
    if (robot_scene->GetPath().empty()) {
        throw std::runtime_error("generated Gobot robot asset has no resource path '" + robot_scene_path + "'");
    }

    auto* scene_root = Object::New<Node3D>();
    try {
        scene_root->SetName(name.has_value() && !name->empty() ? *name : ResourcePathStem(scene_path));
        SetOptionalNodeScript(scene_root, script_path);
        AddMJCFScenePlaneGeoms(scene_root, xml_path);

        Node* robot_instance = robot_scene->Instantiate();
        if (robot_instance == nullptr) {
            throw std::runtime_error("failed to instantiate generated Gobot robot asset '" + robot_scene_path + "'");
        }
        robot_instance->SetName(robot_name);
        robot_instance->SetSceneInstance(robot_scene);
        scene_root->AddChild(robot_instance);

        if (!SaveSceneRoot(scene_root, scene_path)) {
            throw std::runtime_error("failed to save Gobot scene to '" + scene_path + "'");
        }
    } catch (...) {
        Object::Delete(scene_root);
        throw;
    }

    Object::Delete(scene_root);
    return true;
}

void ImportMJCFScene(const std::string& xml_path,
                     const std::string& scene_path,
                     const std::optional<std::string>& name,
                     const std::optional<std::string>& script_path) {
    if (TrySaveSplitMJCFScene(xml_path, scene_path, name, script_path)) {
        return;
    }

    SaveSingleMJCFScene(xml_path, scene_path, name, script_path);
}

PhysicsBackendType ParseBackend(const std::string& backend) {
    if (backend == "null") {
        return PhysicsBackendType::Null;
    }
    if (backend == "mujoco" || backend == "mujoco_cpu") {
        return PhysicsBackendType::MuJoCoCpu;
    }
    if (backend == "mujoco_warp" || backend == "warp") {
        return PhysicsBackendType::MuJoCoWarp;
    }
    throw std::invalid_argument("unknown Gobot physics backend '" + backend + "'");
}

std::string NodeTypeName(const PyNodeHandle& handle) {
    return std::string(handle.Resolve()->GetClassStringName());
}

py::list GetNodeChildren(PyNodeHandle& handle) {
    Node* node = handle.Resolve();
    py::list children;
    for (std::size_t index = 0; index < node->GetChildCount(); ++index) {
        children.append(MakeTypedNodeObject(node->GetChild(static_cast<int>(index)),
                                            PyNodeOwnership::Borrowed,
                                            handle.state ? handle.state->context : nullptr,
                                            handle.state ? handle.state->scene_epoch : 0));
    }
    return children;
}

py::object GetNodeChild(PyNodeHandle& handle, int index) {
    Node* child = handle.Resolve()->GetChild(index);
    if (child == nullptr) {
        throw py::index_error("Gobot node child index is out of range");
    }
    return MakeTypedNodeObject(child,
                               PyNodeOwnership::Borrowed,
                               handle.state ? handle.state->context : nullptr,
                               handle.state ? handle.state->scene_epoch : 0);
}

py::object FindNode(PyNodeHandle& handle, const std::string& path) {
    Node* node = handle.Resolve()->GetNodeOrNull(NodePath(path));
    if (node == nullptr) {
        return py::none();
    }
    return MakeTypedNodeObject(node,
                               PyNodeOwnership::Borrowed,
                               handle.state ? handle.state->context : nullptr,
                               handle.state ? handle.state->scene_epoch : 0);
}

py::object NodeGetProperty(const PyNodeHandle& handle, const std::string& name) {
    const Node* node = handle.Resolve();
    Variant value = node->Get(name);
    if (!value.is_valid()) {
        throw py::key_error("unknown Gobot property '" + name + "'");
    }
    return VariantToPython(value);
}

void NodeSetProperty(PyNodeHandle& handle, const std::string& name, const py::handle& value) {
    Node* node = handle.Resolve();
    ScopedActiveAppContext scoped_context(handle.state ? handle.state->context : nullptr);
    const Property property = node->GetType().get_property(name);
    if (!property.is_valid()) {
        throw py::key_error("unknown Gobot property '" + name + "'");
    }
    Variant variant_value = PythonToVariantForType(value, property.get_type());
    if (IsSceneScriptRuntimeMutation()) {
        if (!node->Set(name, variant_value)) {
            throw std::runtime_error("failed to set Gobot runtime node property '" + name + "'");
        }
    } else {
        ExecuteSetNodeProperty(node, name, std::move(variant_value));
    }
    handle.RefreshSnapshot(node);
}

py::object NodeAddChild(PyNodeHandle& handle, PyNodeHandle& child_handle, bool force_readable_name) {
    Node* node = handle.Resolve();
    Node* child = child_handle.Resolve();
    ScopedActiveAppContext scoped_context(handle.state ? handle.state->context : nullptr);
    if (IsSceneScriptRuntimeMutation()) {
        node->AddChild(child, force_readable_name);
    } else {
        ExecuteSceneMutation("add_child", [&]() {
            return std::make_unique<AddChildNodeCommand>(node->GetInstanceId(),
                                                        child->GetInstanceId(),
                                                        force_readable_name);
        });
    }
    child_handle.TransferToTree();
    child_handle.RefreshSnapshot(child);
    return MakeTypedNodeObject(child,
                               PyNodeOwnership::Borrowed,
                               handle.state ? handle.state->context : nullptr,
                               handle.state ? handle.state->scene_epoch : 0);
}

py::object NodeRemoveChild(PyNodeHandle& handle, PyNodeHandle& child_handle, bool delete_child) {
    Node* node = handle.Resolve();
    Node* child = child_handle.Resolve();
    ScopedActiveAppContext scoped_context(handle.state ? handle.state->context : nullptr);
    if (child->GetParent() != node) {
        throw std::invalid_argument("Gobot node '" + child->GetName() + "' is not a child of '" +
                                    node->GetName() + "'");
    }

    if (delete_child) {
        if (IsSceneScriptRuntimeMutation()) {
            node->RemoveChild(child);
            Object::Delete(child);
        } else {
            ExecuteSceneMutation("remove_child_delete", [&]() {
                return std::make_unique<RemoveChildNodeCommand>(node->GetInstanceId(), child->GetInstanceId(), true);
            });
        }
        child_handle.Invalidate();
        return py::none();
    }

    std::shared_ptr<PyNodeHandleState> detached_state = child_handle.state;
    if (IsSceneScriptRuntimeMutation()) {
        node->RemoveChild(child);
    } else {
        ExecuteSceneMutation("remove_child_detach", [&]() {
            return std::make_unique<RemoveChildNodeCommand>(node->GetInstanceId(), child->GetInstanceId(), false);
        });
    }
    child_handle.TransferToDetachedOwned(child);
    return py::cast(PyNodeHandle(detached_state, ExpectedTypeForNode(child)));
}

void NodeReparent(PyNodeHandle& handle, PyNodeHandle& parent_handle) {
    Node* node = handle.Resolve();
    Node* parent = parent_handle.Resolve();
    ScopedActiveAppContext scoped_context(handle.state ? handle.state->context : nullptr);
    if (IsSceneScriptRuntimeMutation()) {
        node->Reparent(parent);
    } else {
        ExecuteSceneMutation("reparent", [&]() {
            return std::make_unique<ReparentNodeCommand>(node->GetInstanceId(), parent->GetInstanceId());
        });
    }
    handle.TransferToTree();
    handle.RefreshSnapshot(node);
}

py::object NodeGetParent(PyNodeHandle& handle) {
    Node* parent = handle.Resolve()->GetParent();
    if (parent == nullptr) {
        return py::none();
    }
    return MakeTypedNodeObject(parent,
                               PyNodeOwnership::Borrowed,
                               handle.state ? handle.state->context : nullptr,
                               handle.state ? handle.state->scene_epoch : 0);
}

py::list NodeGetPropertyNames(const PyNodeHandle& handle) {
    const Node* node = handle.Resolve();
    py::list names;
    for (const Property& property : node->GetType().get_properties()) {
        names.append(py::str(property.get_name().data()));
    }
    return names;
}

std::string NodeGetName(const PyNodeHandle& handle) {
    return handle.Resolve()->GetName();
}

void NodeSetName(PyNodeHandle& handle, const std::string& name) {
    Node* node = handle.Resolve();
    if (IsSceneScriptRuntimeMutation()) {
        node->SetName(name);
    } else {
        ExecuteSceneMutation("rename_node", [&]() {
            return std::make_unique<RenameNodeCommand>(node->GetInstanceId(), name);
        });
    }
    handle.RefreshSnapshot(node);
}

std::string NodeGetPath(const PyNodeHandle& handle) {
    Node* node = handle.Resolve();
    if (node->IsInsideTree()) {
        return node->GetPath().operator std::string();
    }
    return node->GetName();
}

std::uint64_t NodeGetId(const PyNodeHandle& handle) {
    return handle.state ? static_cast<std::uint64_t>(handle.state->id) : 0;
}

bool NodeIsValid(const PyNodeHandle& handle) {
    try {
        return handle.TryResolve() != nullptr &&
               (!handle.state || handle.state->ownership == PyNodeOwnership::DetachedOwned ||
                handle.state->scene_epoch == ActiveSceneEpoch());
    } catch (...) {
        return false;
    }
}

py::dict NodeToDict(const PyNodeHandle& handle) {
    const Node* node = handle.Resolve();
    py::dict result;
    result["id"] = NodeGetId(handle);
    result["name"] = node->GetName();
    result["path"] = NodeGetPath(handle);
    result["type"] = std::string(node->GetClassStringName());
    py::dict properties;
    for (const Property& property : node->GetType().get_properties()) {
        Variant value = property.get_value(node);
        if (value.is_valid()) {
            properties[py::str(property.get_name().data())] = VariantToPython(value);
        }
    }
    result["properties"] = properties;
    return result;
}

Node* CreateNode(const std::string& type_name, const std::string& name) {
    Node* node = NodeCreationRegistry::CreateNode(type_name);
    if (node == nullptr) {
        throw std::invalid_argument("unknown Gobot node type '" + type_name + "'");
    }
    if (!name.empty()) {
        node->SetName(name);
    }
    return node;
}

py::dict ResourceToPythonDict(const Ref<Resource>& resource) {
    py::dict result;
    if (!resource.IsValid()) {
        return result;
    }

    result["type"] = py::str(resource->GetClassStringName().data());
    result["path"] = resource->GetPath();
    result["name"] = resource->GetName();
    py::dict properties;
    Instance instance(resource.Get());
    const Type type = resource->GetType();
    for (const Property& property : type.get_properties()) {
        Variant value = property.get_value(instance);
        if (!value.is_valid()) {
            continue;
        }
        properties[py::str(property.get_name().data())] = VariantToPython(value);
    }
    result["properties"] = properties;
    for (const auto& item : properties) {
        result[item.first] = item.second;
    }
    return result;
}

py::dict TransformToPythonDict(const Affine3& transform) {
    const Vector3 position = transform.translation();
    const Quaternion rotation(transform.linear());

    py::dict result;
    result["position"] = Vector3ToPython(position);
    result["quaternion"] = QuaternionWxyzToPython(rotation);
    result["matrix"] = Matrix4ToPython(transform.matrix());
    return result;
}

template <typename T>
py::array_t<T> MakeArray(std::vector<T> data, std::vector<py::ssize_t> shape) {
    auto* storage = new std::vector<T>(std::move(data));
    T* pointer = storage->data();
    py::capsule owner(storage, [](void* value) {
        delete static_cast<std::vector<T>*>(value);
    });
    return py::array_t<T>(std::move(shape), pointer, owner);
}

py::array_t<RealType> MakeRealArray(std::vector<RealType> data, std::vector<py::ssize_t> shape) {
    return MakeArray<RealType>(std::move(data), std::move(shape));
}

py::array_t<std::uint8_t> MakeBoolArray(std::vector<std::uint8_t> data, std::vector<py::ssize_t> shape) {
    return MakeArray<std::uint8_t>(std::move(data), std::move(shape));
}

std::vector<double> PythonToFixedDoubleArray(const py::handle& object,
                                             py::ssize_t expected_size,
                                             const std::string& description) {
    if (py::isinstance<py::str>(object) || py::isinstance<py::bytes>(object)) {
        throw std::invalid_argument("expected a " + description);
    }

    py::array_t<double, py::array::c_style | py::array::forcecast> array =
            py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(object);
    if (!array) {
        throw std::invalid_argument("expected a " + description);
    }

    py::buffer_info info = array.request();
    if (info.ndim != 1 || info.shape[0] != expected_size) {
        throw std::invalid_argument("expected a " + description);
    }

    const auto* data = static_cast<const double*>(info.ptr);
    return std::vector<double>(data, data + expected_size);
}

void FillVector3(std::vector<RealType>& values, std::size_t offset, const Vector3& vector) {
    values[offset + 0] = vector.x();
    values[offset + 1] = vector.y();
    values[offset + 2] = vector.z();
}

void FillQuaternionWxyz(std::vector<RealType>& values, std::size_t offset, const Quaternion& quaternion) {
    values[offset + 0] = quaternion.w();
    values[offset + 1] = quaternion.x();
    values[offset + 2] = quaternion.y();
    values[offset + 3] = quaternion.z();
}

Quaternion PythonToQuaternionWxyz(const py::handle& object) {
    std::vector<double> values = PythonToFixedDoubleArray(object,
                                                          4,
                                                          "4-element quaternion in [w, x, y, z] order");
    Quaternion quaternion(static_cast<RealType>(values[0]),
                          static_cast<RealType>(values[1]),
                          static_cast<RealType>(values[2]),
                          static_cast<RealType>(values[3]));
    if (quaternion.norm() <= CMP_EPSILON) {
        return Quaternion::Identity();
    }
    quaternion.normalize();
    return quaternion;
}

Vector2 PythonToVector2(const py::handle& object) {
    std::vector<double> values = PythonToFixedDoubleArray(object, 2, "2-element vector");
    return {static_cast<RealType>(values[0]), static_cast<RealType>(values[1])};
}

Color PythonToColor4(const py::handle& object) {
    py::sequence sequence = py::reinterpret_borrow<py::sequence>(object);
    if (sequence.size() != 4) {
        throw std::invalid_argument("expected a 4-element color");
    }
    return Color(static_cast<float>(py::cast<RealType>(sequence[0])),
                 static_cast<float>(py::cast<RealType>(sequence[1])),
                 static_cast<float>(py::cast<RealType>(sequence[2])),
                 static_cast<float>(py::cast<RealType>(sequence[3])));
}

py::tuple ColorToPython(const Color& color) {
    return py::make_tuple(color.red(), color.green(), color.blue(), color.alpha());
}

std::vector<Vector3> PythonToVector3List(const py::handle& object) {
    py::sequence sequence = py::reinterpret_borrow<py::sequence>(object);
    std::vector<Vector3> values;
    values.reserve(static_cast<std::size_t>(sequence.size()));
    for (py::handle item : sequence) {
        values.push_back(PythonToVector3(item));
    }
    return values;
}

py::list Vector3ListToPython(const std::vector<Vector3>& values) {
    py::list result;
    for (const Vector3& value : values) {
        result.append(Vector3ToPython(value));
    }
    return result;
}

std::vector<DebugArrow> PythonToDebugArrows(const py::handle& object) {
    if (object.is_none()) {
        return {};
    }

    py::sequence sequence = py::reinterpret_borrow<py::sequence>(object);
    std::vector<DebugArrow> arrows;
    arrows.reserve(static_cast<std::size_t>(sequence.size()));
    for (py::handle item : sequence) {
        py::dict values = py::reinterpret_borrow<py::dict>(item);
        DebugArrow arrow;
        if (!values.contains("start")) {
            throw std::invalid_argument("debug arrow is missing 'start'");
        }
        if (!values.contains("vector")) {
            throw std::invalid_argument("debug arrow is missing 'vector'");
        }
        arrow.start = PythonToVector3(values["start"]);
        arrow.vector = PythonToVector3(values["vector"]);
        if (values.contains("color")) {
            arrow.color = PythonToColor4(values["color"]);
        }
        if (values.contains("scale")) {
            arrow.scale = py::cast<RealType>(values["scale"]);
        }
        if (values.contains("label")) {
            arrow.label = py::cast<std::string>(values["label"]);
        }
        arrows.push_back(std::move(arrow));
    }
    return arrows;
}

Affine3 PythonToTransformWxyz(const py::handle& position, const py::handle& orientation) {
    Affine3 transform = Affine3::Identity();
    transform.translation() = PythonToVector3(position);
    transform.linear() = PythonToQuaternionWxyz(orientation).toRotationMatrix();
    return transform;
}

py::array_t<std::uint8_t> CaptureRgb(const py::handle& root_handle,
                                     int width,
                                     int height,
                                     const py::handle& eye,
                                     const py::handle& target,
                                     const py::handle& up,
                                     RealType fov_y,
                                     RealType z_near,
                                     RealType z_far,
                                     const py::handle& debug_arrows) {
    if (width <= 0 || height <= 0) {
        throw std::invalid_argument("capture size must be positive");
    }

    Node* root = nullptr;
    if (root_handle.is_none()) {
        root = ActiveSceneRoot();
    } else {
        const PyNodeHandle& handle = py::cast<const PyNodeHandle&>(root_handle);
        root = handle.Resolve();
    }
    if (root == nullptr) {
        throw std::runtime_error("cannot capture RGB without a scene root");
    }

    if (!RenderServer::HasInstance()) {
        HeadlessRenderContext& headless_context = EnsureHeadlessRenderContext();
        if (!headless_context.Initialize()) {
            throw std::runtime_error(headless_context.GetLastError());
        }
    }

    auto* render_server = RenderServer::GetInstance();
    if (render_server == nullptr) {
        throw std::runtime_error("Gobot RenderServer is not initialized");
    }

    Camera3D camera;
    camera.SetAspect(static_cast<RealType>(width) / static_cast<RealType>(height));
    camera.SetPerspective(fov_y, z_near, z_far);
    camera.SetViewMatrix(PythonToVector3(eye), PythonToVector3(target), PythonToVector3(up));

    RID capture_viewport = render_server->ViewportCreate();
    render_server->ViewportSetSize(capture_viewport, width, height);
    render_server->RenderSceneToViewport(capture_viewport, root, &camera);
    render_server->RenderDebugArrowsToViewport(capture_viewport, &camera, PythonToDebugArrows(debug_arrows));
    std::vector<std::uint8_t> pixels = render_server->ReadViewportRgbPixels(capture_viewport, true);
    render_server->Free(capture_viewport);
    if (pixels.size() != static_cast<std::size_t>(width) * height * 3) {
        throw std::runtime_error("RGB capture readback returned an unexpected pixel buffer size");
    }

    py::array_t<std::uint8_t> array({height, width, 3});
    py::buffer_info info = array.request();
    std::copy(pixels.begin(), pixels.end(), static_cast<std::uint8_t*>(info.ptr));
    return array;
}

std::string PhysicsJointControlModeName(PhysicsJointControlMode mode) {
    switch (mode) {
        case PhysicsJointControlMode::Passive:
            return "passive";
        case PhysicsJointControlMode::Position:
            return "position";
        case PhysicsJointControlMode::Velocity:
            return "velocity";
        case PhysicsJointControlMode::Effort:
            return "effort";
    }

    return "unknown";
}

std::string PhysicsSensorTypeName(PhysicsSensorType type) {
    switch (type) {
        case PhysicsSensorType::IMU:
            return "imu";
        case PhysicsSensorType::AngularMomentum:
            return "angular_momentum";
        case PhysicsSensorType::Contact:
            return "contact";
        case PhysicsSensorType::RayCast:
            return "raycast";
        case PhysicsSensorType::TerrainHeight:
            return "terrain_height";
        case PhysicsSensorType::HeightScanner:
            return "height_scanner";
        case PhysicsSensorType::Unknown:
            break;
    }

    return "unknown";
}

std::string RayReductionModeName(RayReductionMode reduction_mode) {
    switch (reduction_mode) {
        case RayReductionMode::None:
            return "none";
        case RayReductionMode::Min:
            return "min";
        case RayReductionMode::Max:
            return "max";
        case RayReductionMode::Mean:
            return "mean";
    }

    return "none";
}

std::string RayPatternModeName(RayPatternMode pattern_mode) {
    switch (pattern_mode) {
        case RayPatternMode::Custom:
            return "custom";
        case RayPatternMode::Grid:
            return "grid";
    }
    return "custom";
}

std::string RayAlignmentModeName(RayAlignmentMode ray_alignment) {
    switch (ray_alignment) {
        case RayAlignmentMode::World:
            return "world";
        case RayAlignmentMode::Base:
            return "base";
        case RayAlignmentMode::Yaw:
            return "yaw";
    }
    return "world";
}

py::dict JointSnapshotToPythonDict(const PhysicsJointSnapshot& joint) {
    py::dict result;
    result["name"] = joint.name;
    result["parent_link"] = joint.parent_link;
    result["child_link"] = joint.child_link;
    result["type"] = static_cast<JointType>(joint.joint_type);
    result["axis"] = Vector3ToPython(joint.axis);
    result["lower_limit"] = joint.lower_limit;
    result["upper_limit"] = joint.upper_limit;
    result["effort_limit"] = joint.effort_limit;
    result["velocity_limit"] = joint.velocity_limit;
    result["damping"] = joint.damping;
    result["joint_position"] = joint.joint_position;
    result["initial_position"] = joint.initial_position;
    result["drive_mode"] = joint.drive_mode;
    result["drive_stiffness"] = joint.drive_stiffness;
    result["drive_damping"] = joint.drive_damping;
    result["control_lower_limit"] = joint.control_lower_limit;
    result["control_upper_limit"] = joint.control_upper_limit;
    result["force_lower_limit"] = joint.force_lower_limit;
    result["force_upper_limit"] = joint.force_upper_limit;
    result["gear"] = joint.gear;
    result["global_transform"] = TransformToPythonDict(joint.global_transform);
    return result;
}

py::dict LinkSnapshotToPythonDict(const PhysicsLinkSnapshot& link) {
    py::dict result;
    result["name"] = link.name;
    result["role"] = link.role == PhysicsLinkRole::VirtualRoot ? "virtual_root" : "physical";
    result["mass"] = link.mass;
    result["center_of_mass"] = Vector3ToPython(link.center_of_mass);
    result["inertia_diagonal"] = Vector3ToPython(link.inertia_diagonal);
    result["inertia_off_diagonal"] = Vector3ToPython(link.inertia_off_diagonal);
    result["global_transform"] = TransformToPythonDict(link.global_transform);
    result["collision_shape_count"] = link.collision_shapes.size();
    return result;
}

py::dict SensorSnapshotToPythonDict(const PhysicsSensorSnapshot& sensor) {
    py::dict result;
    result["name"] = sensor.name;
    result["sensor_name"] = sensor.name;
    result["link_name"] = sensor.link_name;
    result["type"] = PhysicsSensorTypeName(sensor.type);
    result["enabled"] = sensor.enabled;
    result["sensor_period"] = sensor.sensor_period;
    result["noise_stddev"] = sensor.noise_stddev;
    result["visualize_debug"] = sensor.visualize_debug;
    result["debug_marker_radius"] = sensor.node != nullptr ? sensor.node->GetDebugMarkerRadius() : 0.0;
    result["radius"] = sensor.radius;
    result["min_threshold"] = sensor.min_threshold;
    result["max_threshold"] = sensor.max_threshold;
    result["sample_offsets"] = Vector3ListToPython(sensor.sample_offsets);
    result["ray_direction"] = Vector3ToPython(sensor.ray_direction);
    result["ray_direction_world_space"] = sensor.ray_direction_world_space;
    result["max_distance"] = sensor.max_distance;
    result["reduction_mode"] = RayReductionModeName(sensor.reduction_mode);
    result["pattern_mode"] = RayPatternModeName(sensor.pattern_mode);
    result["grid_size"] = Vector2ToPython(sensor.grid_size);
    result["grid_resolution"] = sensor.grid_resolution;
    result["ray_alignment"] = RayAlignmentModeName(sensor.ray_alignment);
    result["channel_names"] = sensor.channel_names;
    result["global_transform"] = TransformToPythonDict(sensor.global_transform);
    result["local_transform"] = TransformToPythonDict(sensor.local_transform);
    return result;
}

py::dict JointStateToPythonDict(const PhysicsJointState& joint) {
    py::dict result;
    result["robot_name"] = joint.robot_name;
    result["name"] = joint.joint_name;
    result["joint_name"] = joint.joint_name;
    result["type"] = static_cast<JointType>(joint.joint_type);
    result["position"] = joint.position;
    result["velocity"] = joint.velocity;
    result["effort"] = joint.effort;
    result["control_mode"] = PhysicsJointControlModeName(joint.control_mode);
    result["target_position"] = joint.target_position;
    result["target_velocity"] = joint.target_velocity;
    result["target_effort"] = joint.target_effort;
    return result;
}

py::dict LinkStateToPythonDict(const PhysicsLinkState& link) {
    py::dict result;
    result["robot_name"] = link.robot_name;
    result["name"] = link.link_name;
    result["link_name"] = link.link_name;
    result["role"] = link.role == PhysicsLinkRole::VirtualRoot ? "virtual_root" : "physical";
    result["global_transform"] = TransformToPythonDict(link.global_transform);
    result["linear_velocity"] = Vector3ToPython(link.linear_velocity);
    result["angular_velocity"] = Vector3ToPython(link.angular_velocity);
    return result;
}

py::dict ContactStateToPythonDict(const PhysicsContactState& contact) {
    py::dict result;
    result["robot_name"] = contact.robot_name;
    result["link_name"] = contact.link_name;
    result["other_robot_name"] = contact.other_robot_name;
    result["other_link_name"] = contact.other_link_name;
    result["position"] = Vector3ToPython(contact.position);
    result["normal"] = Vector3ToPython(contact.normal);
    result["force"] = Vector3ToPython(contact.force);
    result["normal_force"] = contact.normal_force;
    result["distance"] = contact.distance;
    return result;
}

py::dict SensorRaycastHitToPythonDict(const PhysicsSensorRaycastHit& hit) {
    py::dict result;
    result["hit"] = hit.hit;
    result["origin"] = Vector3ToPython(hit.origin);
    result["point"] = Vector3ToPython(hit.point);
    result["normal"] = Vector3ToPython(hit.normal);
    result["distance"] = hit.distance;
    result["terrain_name"] = hit.terrain_name;
    return result;
}

py::dict SensorStateToPythonDict(const PhysicsSensorState& sensor) {
    py::dict result;
    result["robot_name"] = sensor.robot_name;
    result["link_name"] = sensor.link_name;
    result["name"] = sensor.sensor_name;
    result["sensor_name"] = sensor.sensor_name;
    result["type"] = PhysicsSensorTypeName(sensor.type);
    result["enabled"] = sensor.enabled;
    result["visualize_debug"] = sensor.visualize_debug;
    result["debug_marker_radius"] = sensor.node != nullptr ? sensor.node->GetDebugMarkerRadius() : 0.0;
    result["global_transform"] = TransformToPythonDict(sensor.global_transform);
    result["values"] = sensor.values;
    py::list hits;
    for (const PhysicsSensorRaycastHit& hit : sensor.hits) {
        hits.append(SensorRaycastHitToPythonDict(hit));
    }
    result["hits"] = hits;
    result["channel_names"] = sensor.channel_names;
    result["timestamp"] = sensor.timestamp;
    return result;
}

py::dict RuntimeStateToPythonDict(const PhysicsSceneState& state) {
    py::dict result;
    py::list robots;
    for (const PhysicsRobotState& robot : state.robots) {
        py::dict robot_dict;
        robot_dict["name"] = robot.name;

        py::list links;
        for (const PhysicsLinkState& link : robot.links) {
            links.append(LinkStateToPythonDict(link));
        }
        robot_dict["links"] = links;

        py::list joints;
        for (const PhysicsJointState& joint : robot.joints) {
            joints.append(JointStateToPythonDict(joint));
        }
        robot_dict["joints"] = joints;

        py::list sensors;
        for (const PhysicsSensorState& sensor : robot.sensors) {
            sensors.append(SensorStateToPythonDict(sensor));
        }
        robot_dict["sensors"] = sensors;
        robots.append(robot_dict);
    }

    py::list contacts;
    for (const PhysicsContactState& contact : state.contacts) {
        contacts.append(ContactStateToPythonDict(contact));
    }

    py::list sensors;
    for (const PhysicsSensorState& sensor : state.loose_sensors) {
        sensors.append(SensorStateToPythonDict(sensor));
    }

    result["robots"] = robots;
    result["contacts"] = contacts;
    result["sensors"] = sensors;
    result["total_link_count"] = state.total_link_count;
    result["total_joint_count"] = state.total_joint_count;
    result["total_sensor_count"] = state.total_sensor_count;
    return result;
}

const PhysicsRobotSnapshot* FindRobotSnapshot(const PhysicsSceneSnapshot& snapshot,
                                              const std::string& robot_name) {
    for (const PhysicsRobotSnapshot& robot : snapshot.robots) {
        if (robot.name == robot_name) {
            return &robot;
        }
    }
    return nullptr;
}

const PhysicsRobotState* FindRobotState(const PhysicsSceneState& state,
                                        const std::string& robot_name) {
    for (const PhysicsRobotState& robot : state.robots) {
        if (robot.name == robot_name) {
            return &robot;
        }
    }
    return nullptr;
}

const PhysicsLinkState* FindLinkState(const PhysicsRobotState& robot,
                                      const std::string& link_name) {
    for (const PhysicsLinkState& link : robot.links) {
        if (link.link_name == link_name) {
            return &link;
        }
    }
    return nullptr;
}

const PhysicsJointState* FindJointState(const PhysicsRobotState& robot,
                                        const std::string& joint_name) {
    for (const PhysicsJointState& joint : robot.joints) {
        if (joint.joint_name == joint_name) {
            return &joint;
        }
    }
    return nullptr;
}

const PhysicsJointSnapshot* FindJointSnapshot(const PhysicsRobotSnapshot& robot,
                                              const std::string& joint_name) {
    for (const PhysicsJointSnapshot& joint : robot.joints) {
        if (joint.name == joint_name) {
            return &joint;
        }
    }
    return nullptr;
}

const PhysicsSensorState* FindSensorState(const PhysicsRobotState& robot,
                                          const std::string& sensor_name) {
    for (const PhysicsSensorState& sensor : robot.sensors) {
        if (sensor.sensor_name == sensor_name) {
            return &sensor;
        }
    }
    return nullptr;
}

const PhysicsSensorSnapshot* FindSensorSnapshot(const PhysicsRobotSnapshot& robot,
                                                const std::string& sensor_name) {
    for (const PhysicsSensorSnapshot& sensor : robot.sensors) {
        if (sensor.name == sensor_name) {
            return &sensor;
        }
    }
    return nullptr;
}

SimulationServer* SimulationServerForRobotHandle(const PyRobot3DHandle& handle) {
    EngineContext* context = ResolveHandleContext(handle.state ? handle.state->context : nullptr);
    if (context == nullptr) {
        throw std::runtime_error("Gobot robot node is not associated with an app context");
    }

    SimulationServer* simulation = context->GetSimulationServer();
    if (simulation == nullptr) {
        throw std::runtime_error("Gobot robot node app context has no SimulationServer");
    }

    return simulation;
}

SimulationScene* RuntimeSceneForRobotHandle(const PyRobot3DHandle& handle) {
    Robot3D* robot = handle.ResolveAs<Robot3D>();
    SimulationServer* simulation = SimulationServerForRobotHandle(handle);
    SimulationScene* runtime_scene = simulation->GetRuntimeScene();
    if (runtime_scene == nullptr || !runtime_scene->IsValid()) {
        throw std::runtime_error("simulation runtime scene has not been built");
    }

    const Node* runtime_root = runtime_scene->GetSceneRoot();
    if (runtime_root == nullptr) {
        throw std::runtime_error("simulation runtime scene has no scene root");
    }
    if (robot != runtime_root && !runtime_root->IsAncestorOf(robot)) {
        throw std::runtime_error("Gobot robot node '" + robot->GetName() +
                                 "' is not part of the active runtime scene");
    }

    return runtime_scene;
}

Robot3D* RuntimeRobotForNodeHandle(const PyNodeHandle& handle) {
    Node* node = handle.Resolve();
    for (Node* current = node; current != nullptr; current = current->GetParent()) {
        if (auto* robot = Object::PointerCastTo<Robot3D>(current)) {
            return robot;
        }
    }
    throw std::runtime_error("Gobot runtime node '" + node->GetName() + "' is not under a Robot3D node");
}

SimulationScene* RuntimeSceneForNodeHandle(const PyNodeHandle& handle) {
    Robot3D* robot = RuntimeRobotForNodeHandle(handle);
    const PyRobot3DHandle robot_handle(robot,
                                       "Robot3D",
                                       handle.state ? handle.state->context : nullptr,
                                       handle.state ? handle.state->scene_epoch : 0,
                                       PyNodeOwnership::Borrowed);
    return RuntimeSceneForRobotHandle(robot_handle);
}

Ref<PhysicsWorld> RuntimeWorldForRobotHandle(const PyRobot3DHandle& handle) {
    SimulationServer* simulation = SimulationServerForRobotHandle(handle);
    Ref<PhysicsWorld> world = simulation->GetWorld();
    if (!world.IsValid()) {
        throw std::runtime_error("simulation world has not been built from a scene");
    }
    return world;
}

py::dict RobotSnapshotToPythonDict(const PhysicsRobotSnapshot& robot) {
    py::dict result;
    result["name"] = robot.name;
    result["source_path"] = robot.source_path;

    py::list links;
    py::list link_names;
    for (const PhysicsLinkSnapshot& link : robot.links) {
        links.append(LinkSnapshotToPythonDict(link));
        link_names.append(link.name);
    }
    result["links"] = links;
    result["link_names"] = link_names;

    py::list joints;
    py::list joint_names;
    py::list controllable_joint_names;
    for (const PhysicsJointSnapshot& joint : robot.joints) {
        joints.append(JointSnapshotToPythonDict(joint));
        joint_names.append(joint.name);

        const auto joint_type = static_cast<JointType>(joint.joint_type);
        if (joint_type == JointType::Revolute ||
            joint_type == JointType::Continuous ||
            joint_type == JointType::Prismatic) {
            controllable_joint_names.append(joint.name);
        }
    }
    result["joints"] = joints;
    result["joint_names"] = joint_names;
    result["controllable_joint_names"] = controllable_joint_names;

    py::list sensors;
    py::list sensor_names;
    for (const PhysicsSensorSnapshot& sensor : robot.sensors) {
        sensors.append(SensorSnapshotToPythonDict(sensor));
        sensor_names.append(sensor.name);
    }
    result["sensors"] = sensors;
    result["sensor_names"] = sensor_names;
    return result;
}

py::dict RobotStateToPythonDict(const PhysicsRobotState& robot,
                                const PhysicsSceneState* scene_state) {
    py::dict result;
    result["name"] = robot.name;

    py::list links;
    for (const PhysicsLinkState& link : robot.links) {
        links.append(LinkStateToPythonDict(link));
    }
    result["links"] = links;

    py::list joints;
    for (const PhysicsJointState& joint : robot.joints) {
        joints.append(JointStateToPythonDict(joint));
    }
    result["joints"] = joints;

    py::list sensors;
    for (const PhysicsSensorState& sensor : robot.sensors) {
        sensors.append(SensorStateToPythonDict(sensor));
    }
    result["sensors"] = sensors;

    py::list contacts;
    if (scene_state != nullptr) {
        for (const PhysicsContactState& contact : scene_state->contacts) {
            if (contact.robot_name == robot.name || contact.other_robot_name == robot.name) {
                contacts.append(ContactStateToPythonDict(contact));
            }
        }
    }
    result["contacts"] = contacts;
    return result;
}

const PhysicsRobotSnapshot& RequiredRobotSnapshotForHandle(const PyRobot3DHandle& handle) {
    Robot3D* robot = handle.ResolveAs<Robot3D>();
    const PhysicsSceneSnapshot& snapshot = RuntimeWorldForRobotHandle(handle)->GetSceneSnapshot();
    const PhysicsRobotSnapshot* robot_snapshot = FindRobotSnapshot(snapshot, robot->GetName());
    if (robot_snapshot == nullptr) {
        throw std::runtime_error("Gobot runtime snapshot has no robot '" + robot->GetName() + "'");
    }
    return *robot_snapshot;
}

const PhysicsRobotState& RequiredRobotStateForHandle(const PyRobot3DHandle& handle) {
    Robot3D* robot = handle.ResolveAs<Robot3D>();
    const PhysicsSceneState& state = RuntimeWorldForRobotHandle(handle)->GetSceneState();
    const PhysicsRobotState* robot_state = FindRobotState(state, robot->GetName());
    if (robot_state == nullptr) {
        throw std::runtime_error("Gobot runtime state has no robot '" + robot->GetName() + "'");
    }
    return *robot_state;
}

const PhysicsRobotState& RequiredRobotStateForNodeHandle(const PyNodeHandle& handle) {
    Robot3D* robot = RuntimeRobotForNodeHandle(handle);
    const PyRobot3DHandle robot_handle(robot,
                                       "Robot3D",
                                       handle.state ? handle.state->context : nullptr,
                                       handle.state ? handle.state->scene_epoch : 0,
                                       PyNodeOwnership::Borrowed);
    return RequiredRobotStateForHandle(robot_handle);
}

const PhysicsSceneState& RequiredSceneStateForHandle(const PyRobot3DHandle& handle) {
    return RuntimeWorldForRobotHandle(handle)->GetSceneState();
}

py::dict BatchRobotStateToPythonDict(SimulationServer& simulation,
                                     const std::string& robot_name,
                                     const std::string& base_link,
                                     const std::vector<std::string>& joint_names,
                                     const std::vector<std::string>& link_names,
                                     const std::vector<std::string>& sensor_names) {
    Ref<PhysicsWorld> world = simulation.GetWorld();
    if (!world.IsValid()) {
        throw std::runtime_error("simulation world has not been built from a scene");
    }

    const std::size_t environment_count = simulation.GetEnvironmentCount();
    if (environment_count == 0) {
        throw std::runtime_error("simulation environment batch has not been configured");
    }

    const PhysicsSceneSnapshot& snapshot = world->GetSceneSnapshot();
    const PhysicsRobotSnapshot* robot_snapshot = FindRobotSnapshot(snapshot, robot_name);
    if (robot_snapshot == nullptr) {
        throw std::runtime_error("Gobot runtime snapshot has no robot '" + robot_name + "'");
    }

    std::vector<std::string> resolved_link_names = link_names;
    if (resolved_link_names.empty()) {
        resolved_link_names.reserve(robot_snapshot->links.size());
        for (const PhysicsLinkSnapshot& link : robot_snapshot->links) {
            resolved_link_names.push_back(link.name);
        }
    }

    std::unordered_map<std::string, std::int32_t> link_index_by_name;
    for (std::size_t index = 0; index < resolved_link_names.size(); ++index) {
        link_index_by_name[resolved_link_names[index]] = static_cast<std::int32_t>(index);
    }

    const std::size_t joint_count = joint_names.size();
    const std::size_t link_count = resolved_link_names.size();
    const std::size_t sensor_count = sensor_names.size();

    std::vector<RealType> base_position(environment_count * 3, 0.0);
    std::vector<RealType> base_quaternion(environment_count * 4, 0.0);
    std::vector<RealType> base_linear_velocity(environment_count * 3, 0.0);
    std::vector<RealType> base_angular_velocity(environment_count * 3, 0.0);
    std::vector<RealType> joint_position(environment_count * joint_count, 0.0);
    std::vector<RealType> joint_velocity(environment_count * joint_count, 0.0);
    std::vector<RealType> joint_effort(environment_count * joint_count, 0.0);
    std::vector<RealType> joint_target_position(environment_count * joint_count, 0.0);
    std::vector<RealType> joint_target_velocity(environment_count * joint_count, 0.0);
    std::vector<RealType> joint_target_effort(environment_count * joint_count, 0.0);
    std::vector<RealType> joint_lower_limit(joint_count, 0.0);
    std::vector<RealType> joint_upper_limit(joint_count, 0.0);
    std::vector<RealType> link_position(environment_count * link_count * 3, 0.0);
    std::vector<RealType> link_quaternion(environment_count * link_count * 4, 0.0);
    std::vector<RealType> link_linear_velocity(environment_count * link_count * 3, 0.0);
    std::vector<RealType> link_angular_velocity(environment_count * link_count * 3, 0.0);

    std::vector<std::int32_t> sensor_value_count(sensor_count, 0);
    std::vector<std::int32_t> sensor_hit_count(sensor_count, 0);
    std::size_t max_sensor_values = 0;
    std::size_t max_sensor_hits = 0;
    for (std::size_t sensor_index = 0; sensor_index < sensor_count; ++sensor_index) {
        const PhysicsSensorSnapshot* sensor_snapshot = FindSensorSnapshot(*robot_snapshot, sensor_names[sensor_index]);
        if (sensor_snapshot == nullptr) {
            throw std::runtime_error("Gobot runtime snapshot robot '" + robot_name +
                                     "' has no sensor '" + sensor_names[sensor_index] + "'");
        }
        sensor_value_count[sensor_index] = static_cast<std::int32_t>(sensor_snapshot->channel_names.size());
        sensor_hit_count[sensor_index] = static_cast<std::int32_t>(sensor_snapshot->sample_offsets.size());
        max_sensor_values = std::max(max_sensor_values, sensor_snapshot->channel_names.size());
        max_sensor_hits = std::max(max_sensor_hits, sensor_snapshot->sample_offsets.size());
    }

    for (std::size_t joint_index = 0; joint_index < joint_count; ++joint_index) {
        const PhysicsJointSnapshot* joint_snapshot = FindJointSnapshot(*robot_snapshot, joint_names[joint_index]);
        if (joint_snapshot == nullptr) {
            throw std::runtime_error("Gobot runtime snapshot robot '" + robot_name +
                                     "' has no joint '" + joint_names[joint_index] + "'");
        }
        joint_lower_limit[joint_index] = joint_snapshot->lower_limit;
        joint_upper_limit[joint_index] = joint_snapshot->upper_limit;
    }

    std::vector<RealType> sensor_position(environment_count * sensor_count * 3, 0.0);
    std::vector<RealType> sensor_quaternion(environment_count * sensor_count * 4, 0.0);
    std::vector<RealType> sensor_values(environment_count * sensor_count * max_sensor_values, 0.0);
    std::vector<std::uint8_t> sensor_hit(environment_count * sensor_count * max_sensor_hits, 0);
    std::vector<RealType> sensor_hit_origin(environment_count * sensor_count * max_sensor_hits * 3, 0.0);
    std::vector<RealType> sensor_hit_point(environment_count * sensor_count * max_sensor_hits * 3, 0.0);
    std::vector<RealType> sensor_hit_normal(environment_count * sensor_count * max_sensor_hits * 3, 0.0);
    std::vector<RealType> sensor_hit_distance(environment_count * sensor_count * max_sensor_hits, 0.0);

    std::vector<std::size_t> robot_contact_counts(environment_count, 0);
    std::size_t max_contact_count = 0;
    for (std::size_t environment_index = 0; environment_index < environment_count; ++environment_index) {
        const PhysicsSceneState* state = simulation.GetEnvironmentState(environment_index);
        if (state == nullptr) {
            throw std::runtime_error(fmt::format("simulation environment state {} is not available", environment_index));
        }
        for (const PhysicsContactState& contact : state->contacts) {
            if (contact.robot_name == robot_name || contact.other_robot_name == robot_name) {
                ++robot_contact_counts[environment_index];
            }
        }
        max_contact_count = std::max(max_contact_count, robot_contact_counts[environment_index]);
    }

    std::vector<std::int32_t> contact_count(environment_count, 0);
    std::vector<std::int32_t> contact_link_index(environment_count * max_contact_count * 2, -1);
    std::vector<RealType> contact_position(environment_count * max_contact_count * 3, 0.0);
    std::vector<RealType> contact_normal(environment_count * max_contact_count * 3, 0.0);
    std::vector<RealType> contact_force(environment_count * max_contact_count * 3, 0.0);
    std::vector<RealType> contact_normal_force(environment_count * max_contact_count, 0.0);
    std::vector<RealType> contact_distance(environment_count * max_contact_count, 0.0);

    for (std::size_t environment_index = 0; environment_index < environment_count; ++environment_index) {
        const PhysicsSceneState* state = simulation.GetEnvironmentState(environment_index);
        const PhysicsRobotState* robot_state = state != nullptr ? FindRobotState(*state, robot_name) : nullptr;
        if (robot_state == nullptr) {
            throw std::runtime_error("Gobot runtime state has no robot '" + robot_name + "'");
        }

        const PhysicsLinkState* base_state = FindLinkState(*robot_state, base_link);
        if (base_state == nullptr) {
            throw std::runtime_error("Gobot runtime state robot '" + robot_name +
                                     "' has no base link '" + base_link + "'");
        }
        const std::size_t base3 = environment_index * 3;
        const std::size_t base4 = environment_index * 4;
        FillVector3(base_position, base3, base_state->global_transform.translation());
        FillQuaternionWxyz(base_quaternion, base4, Quaternion(base_state->global_transform.linear()));
        FillVector3(base_linear_velocity, base3, base_state->linear_velocity);
        FillVector3(base_angular_velocity, base3, base_state->angular_velocity);

        for (std::size_t joint_index = 0; joint_index < joint_count; ++joint_index) {
            const PhysicsJointState* joint_state = FindJointState(*robot_state, joint_names[joint_index]);
            if (joint_state == nullptr) {
                throw std::runtime_error("Gobot runtime state robot '" + robot_name +
                                         "' has no joint '" + joint_names[joint_index] + "'");
            }
            const std::size_t offset = environment_index * joint_count + joint_index;
            joint_position[offset] = joint_state->position;
            joint_velocity[offset] = joint_state->velocity;
            joint_effort[offset] = joint_state->effort;
            joint_target_position[offset] = joint_state->target_position;
            joint_target_velocity[offset] = joint_state->target_velocity;
            joint_target_effort[offset] = joint_state->target_effort;
        }

        for (std::size_t link_index = 0; link_index < link_count; ++link_index) {
            const PhysicsLinkState* link_state = FindLinkState(*robot_state, resolved_link_names[link_index]);
            if (link_state == nullptr) {
                throw std::runtime_error("Gobot runtime state robot '" + robot_name +
                                         "' has no link '" + resolved_link_names[link_index] + "'");
            }
            const std::size_t offset3 = (environment_index * link_count + link_index) * 3;
            const std::size_t offset4 = (environment_index * link_count + link_index) * 4;
            FillVector3(link_position, offset3, link_state->global_transform.translation());
            FillQuaternionWxyz(link_quaternion, offset4, Quaternion(link_state->global_transform.linear()));
            FillVector3(link_linear_velocity, offset3, link_state->linear_velocity);
            FillVector3(link_angular_velocity, offset3, link_state->angular_velocity);
        }

        for (std::size_t sensor_index = 0; sensor_index < sensor_count; ++sensor_index) {
            const PhysicsSensorState* sensor_state = FindSensorState(*robot_state, sensor_names[sensor_index]);
            if (sensor_state == nullptr) {
                throw std::runtime_error("Gobot runtime state robot '" + robot_name +
                                         "' has no sensor '" + sensor_names[sensor_index] + "'");
            }
            const std::size_t offset3 = (environment_index * sensor_count + sensor_index) * 3;
            const std::size_t offset4 = (environment_index * sensor_count + sensor_index) * 4;
            FillVector3(sensor_position, offset3, sensor_state->global_transform.translation());
            FillQuaternionWxyz(sensor_quaternion, offset4, Quaternion(sensor_state->global_transform.linear()));
            for (std::size_t value_index = 0; value_index < sensor_state->values.size() &&
                                              value_index < max_sensor_values; ++value_index) {
                sensor_values[(environment_index * sensor_count + sensor_index) * max_sensor_values + value_index] =
                        sensor_state->values[value_index];
            }
            for (std::size_t hit_index = 0; hit_index < sensor_state->hits.size() &&
                                            hit_index < max_sensor_hits; ++hit_index) {
                const PhysicsSensorRaycastHit& hit = sensor_state->hits[hit_index];
                const std::size_t hit_base = (environment_index * sensor_count + sensor_index) * max_sensor_hits +
                                             hit_index;
                sensor_hit[hit_base] = hit.hit ? 1 : 0;
                sensor_hit_distance[hit_base] = hit.distance;
                FillVector3(sensor_hit_origin, hit_base * 3, hit.origin);
                FillVector3(sensor_hit_point, hit_base * 3, hit.point);
                FillVector3(sensor_hit_normal, hit_base * 3, hit.normal);
            }
        }

        std::size_t contact_index = 0;
        for (const PhysicsContactState& contact : state->contacts) {
            if (contact.robot_name != robot_name && contact.other_robot_name != robot_name) {
                continue;
            }
            if (contact_index >= max_contact_count) {
                break;
            }
            const std::size_t contact_base = environment_index * max_contact_count + contact_index;
            if (contact.robot_name == robot_name) {
                const auto iter = link_index_by_name.find(contact.link_name);
                if (iter != link_index_by_name.end()) {
                    contact_link_index[contact_base * 2 + 0] = iter->second;
                }
            }
            if (contact.other_robot_name == robot_name) {
                const auto iter = link_index_by_name.find(contact.other_link_name);
                if (iter != link_index_by_name.end()) {
                    contact_link_index[contact_base * 2 + 1] = iter->second;
                }
            }
            FillVector3(contact_position, contact_base * 3, contact.position);
            FillVector3(contact_normal, contact_base * 3, contact.normal);
            FillVector3(contact_force, contact_base * 3, contact.force);
            contact_normal_force[contact_base] = contact.normal_force;
            contact_distance[contact_base] = contact.distance;
            ++contact_index;
        }
        contact_count[environment_index] = static_cast<std::int32_t>(contact_index);
    }

    py::dict result;
    result["robot_name"] = robot_name;
    result["base_link"] = base_link;
    result["joint_names"] = joint_names;
    result["link_names"] = resolved_link_names;
    result["sensor_names"] = sensor_names;
    result["env_count"] = environment_count;
    result["base_position"] = MakeRealArray(std::move(base_position), {static_cast<py::ssize_t>(environment_count), 3});
    result["base_quaternion"] = MakeRealArray(std::move(base_quaternion), {static_cast<py::ssize_t>(environment_count), 4});
    result["base_linear_velocity"] = MakeRealArray(std::move(base_linear_velocity), {static_cast<py::ssize_t>(environment_count), 3});
    result["base_angular_velocity"] = MakeRealArray(std::move(base_angular_velocity), {static_cast<py::ssize_t>(environment_count), 3});
    result["joint_position"] = MakeRealArray(std::move(joint_position), {static_cast<py::ssize_t>(environment_count), static_cast<py::ssize_t>(joint_count)});
    result["joint_velocity"] = MakeRealArray(std::move(joint_velocity), {static_cast<py::ssize_t>(environment_count), static_cast<py::ssize_t>(joint_count)});
    result["joint_effort"] = MakeRealArray(std::move(joint_effort), {static_cast<py::ssize_t>(environment_count), static_cast<py::ssize_t>(joint_count)});
    result["joint_target_position"] = MakeRealArray(std::move(joint_target_position), {static_cast<py::ssize_t>(environment_count), static_cast<py::ssize_t>(joint_count)});
    result["joint_target_velocity"] = MakeRealArray(std::move(joint_target_velocity), {static_cast<py::ssize_t>(environment_count), static_cast<py::ssize_t>(joint_count)});
    result["joint_target_effort"] = MakeRealArray(std::move(joint_target_effort), {static_cast<py::ssize_t>(environment_count), static_cast<py::ssize_t>(joint_count)});
    result["joint_lower_limit"] = MakeRealArray(std::move(joint_lower_limit), {static_cast<py::ssize_t>(joint_count)});
    result["joint_upper_limit"] = MakeRealArray(std::move(joint_upper_limit), {static_cast<py::ssize_t>(joint_count)});
    result["link_position"] = MakeRealArray(std::move(link_position), {static_cast<py::ssize_t>(environment_count), static_cast<py::ssize_t>(link_count), 3});
    result["link_quaternion"] = MakeRealArray(std::move(link_quaternion), {static_cast<py::ssize_t>(environment_count), static_cast<py::ssize_t>(link_count), 4});
    result["link_linear_velocity"] = MakeRealArray(std::move(link_linear_velocity), {static_cast<py::ssize_t>(environment_count), static_cast<py::ssize_t>(link_count), 3});
    result["link_angular_velocity"] = MakeRealArray(std::move(link_angular_velocity), {static_cast<py::ssize_t>(environment_count), static_cast<py::ssize_t>(link_count), 3});
    result["sensor_value_count"] = MakeArray<std::int32_t>(std::move(sensor_value_count), {static_cast<py::ssize_t>(sensor_count)});
    result["sensor_hit_count"] = MakeArray<std::int32_t>(std::move(sensor_hit_count), {static_cast<py::ssize_t>(sensor_count)});
    result["sensor_position"] = MakeRealArray(std::move(sensor_position), {static_cast<py::ssize_t>(environment_count), static_cast<py::ssize_t>(sensor_count), 3});
    result["sensor_quaternion"] = MakeRealArray(std::move(sensor_quaternion), {static_cast<py::ssize_t>(environment_count), static_cast<py::ssize_t>(sensor_count), 4});
    result["sensor_values"] = MakeRealArray(std::move(sensor_values), {static_cast<py::ssize_t>(environment_count), static_cast<py::ssize_t>(sensor_count), static_cast<py::ssize_t>(max_sensor_values)});
    result["sensor_hit"] = MakeBoolArray(std::move(sensor_hit), {static_cast<py::ssize_t>(environment_count), static_cast<py::ssize_t>(sensor_count), static_cast<py::ssize_t>(max_sensor_hits)});
    result["sensor_hit_origin"] = MakeRealArray(std::move(sensor_hit_origin), {static_cast<py::ssize_t>(environment_count), static_cast<py::ssize_t>(sensor_count), static_cast<py::ssize_t>(max_sensor_hits), 3});
    result["sensor_hit_point"] = MakeRealArray(std::move(sensor_hit_point), {static_cast<py::ssize_t>(environment_count), static_cast<py::ssize_t>(sensor_count), static_cast<py::ssize_t>(max_sensor_hits), 3});
    result["sensor_hit_normal"] = MakeRealArray(std::move(sensor_hit_normal), {static_cast<py::ssize_t>(environment_count), static_cast<py::ssize_t>(sensor_count), static_cast<py::ssize_t>(max_sensor_hits), 3});
    result["sensor_hit_distance"] = MakeRealArray(std::move(sensor_hit_distance), {static_cast<py::ssize_t>(environment_count), static_cast<py::ssize_t>(sensor_count), static_cast<py::ssize_t>(max_sensor_hits)});
    result["contact_count"] = MakeArray<std::int32_t>(std::move(contact_count), {static_cast<py::ssize_t>(environment_count)});
    result["contact_link_index"] = MakeArray<std::int32_t>(std::move(contact_link_index), {static_cast<py::ssize_t>(environment_count), static_cast<py::ssize_t>(max_contact_count), 2});
    result["contact_position"] = MakeRealArray(std::move(contact_position), {static_cast<py::ssize_t>(environment_count), static_cast<py::ssize_t>(max_contact_count), 3});
    result["contact_normal"] = MakeRealArray(std::move(contact_normal), {static_cast<py::ssize_t>(environment_count), static_cast<py::ssize_t>(max_contact_count), 3});
    result["contact_force"] = MakeRealArray(std::move(contact_force), {static_cast<py::ssize_t>(environment_count), static_cast<py::ssize_t>(max_contact_count), 3});
    result["contact_normal_force"] = MakeRealArray(std::move(contact_normal_force), {static_cast<py::ssize_t>(environment_count), static_cast<py::ssize_t>(max_contact_count)});
    result["contact_distance"] = MakeRealArray(std::move(contact_distance), {static_cast<py::ssize_t>(environment_count), static_cast<py::ssize_t>(max_contact_count)});
    return result;
}

py::dict BatchRobotStateArraysToPythonDict(MuJoCoPhysicsWorld::BatchRobotStateArrays arrays) {
    const auto environment_count = static_cast<py::ssize_t>(arrays.environment_count);
    const auto joint_count = static_cast<py::ssize_t>(arrays.joint_names.size());
    const auto link_count = static_cast<py::ssize_t>(arrays.link_names.size());
    const auto sensor_count = static_cast<py::ssize_t>(arrays.sensor_names.size());
    const auto max_sensor_values = static_cast<py::ssize_t>(arrays.max_sensor_values);
    const auto max_sensor_hits = static_cast<py::ssize_t>(arrays.max_sensor_hits);
    const auto max_contact_count = static_cast<py::ssize_t>(arrays.max_contact_count);

    py::dict result;
    result["robot_name"] = arrays.robot_name;
    result["base_link"] = arrays.base_link;
    result["joint_names"] = arrays.joint_names;
    result["link_names"] = arrays.link_names;
    result["sensor_names"] = arrays.sensor_names;
    result["env_count"] = arrays.environment_count;
    result["base_position"] = MakeRealArray(std::move(arrays.base_position), {environment_count, 3});
    result["base_quaternion"] = MakeRealArray(std::move(arrays.base_quaternion), {environment_count, 4});
    result["base_linear_velocity"] = MakeRealArray(std::move(arrays.base_linear_velocity), {environment_count, 3});
    result["base_angular_velocity"] = MakeRealArray(std::move(arrays.base_angular_velocity), {environment_count, 3});
    result["joint_position"] = MakeRealArray(std::move(arrays.joint_position), {environment_count, joint_count});
    result["joint_velocity"] = MakeRealArray(std::move(arrays.joint_velocity), {environment_count, joint_count});
    result["joint_effort"] = MakeRealArray(std::move(arrays.joint_effort), {environment_count, joint_count});
    result["joint_target_position"] = MakeRealArray(std::move(arrays.joint_target_position), {environment_count, joint_count});
    result["joint_target_velocity"] = MakeRealArray(std::move(arrays.joint_target_velocity), {environment_count, joint_count});
    result["joint_target_effort"] = MakeRealArray(std::move(arrays.joint_target_effort), {environment_count, joint_count});
    result["joint_lower_limit"] = MakeRealArray(std::move(arrays.joint_lower_limit), {joint_count});
    result["joint_upper_limit"] = MakeRealArray(std::move(arrays.joint_upper_limit), {joint_count});
    result["link_position"] = MakeRealArray(std::move(arrays.link_position), {environment_count, link_count, 3});
    result["link_quaternion"] = MakeRealArray(std::move(arrays.link_quaternion), {environment_count, link_count, 4});
    result["link_linear_velocity"] = MakeRealArray(std::move(arrays.link_linear_velocity), {environment_count, link_count, 3});
    result["link_angular_velocity"] = MakeRealArray(std::move(arrays.link_angular_velocity), {environment_count, link_count, 3});
    result["sensor_value_count"] = MakeArray<std::int32_t>(std::move(arrays.sensor_value_count), {sensor_count});
    result["sensor_hit_count"] = MakeArray<std::int32_t>(std::move(arrays.sensor_hit_count), {sensor_count});
    result["sensor_position"] = MakeRealArray(std::move(arrays.sensor_position), {environment_count, sensor_count, 3});
    result["sensor_quaternion"] = MakeRealArray(std::move(arrays.sensor_quaternion), {environment_count, sensor_count, 4});
    result["sensor_values"] = MakeRealArray(std::move(arrays.sensor_values), {environment_count, sensor_count, max_sensor_values});
    result["sensor_hit"] = MakeBoolArray(std::move(arrays.sensor_hit), {environment_count, sensor_count, max_sensor_hits});
    result["sensor_hit_origin"] = MakeRealArray(std::move(arrays.sensor_hit_origin), {environment_count, sensor_count, max_sensor_hits, 3});
    result["sensor_hit_point"] = MakeRealArray(std::move(arrays.sensor_hit_point), {environment_count, sensor_count, max_sensor_hits, 3});
    result["sensor_hit_normal"] = MakeRealArray(std::move(arrays.sensor_hit_normal), {environment_count, sensor_count, max_sensor_hits, 3});
    result["sensor_hit_distance"] = MakeRealArray(std::move(arrays.sensor_hit_distance), {environment_count, sensor_count, max_sensor_hits});
    result["contact_count"] = MakeArray<std::int32_t>(std::move(arrays.contact_count), {environment_count});
    result["contact_link_index"] = MakeArray<std::int32_t>(std::move(arrays.contact_link_index), {environment_count, max_contact_count, 2});
    result["contact_position"] = MakeRealArray(std::move(arrays.contact_position), {environment_count, max_contact_count, 3});
    result["contact_normal"] = MakeRealArray(std::move(arrays.contact_normal), {environment_count, max_contact_count, 3});
    result["contact_force"] = MakeRealArray(std::move(arrays.contact_force), {environment_count, max_contact_count, 3});
    result["contact_normal_force"] = MakeRealArray(std::move(arrays.contact_normal_force), {environment_count, max_contact_count});
    result["contact_distance"] = MakeRealArray(std::move(arrays.contact_distance), {environment_count, max_contact_count});
    return result;
}


EngineContext& EnsureRuntimeContext() {
    return GetActiveAppContext();
}

Node* PyNodeHandle::Resolve() const {
    if (!state || state->id.IsNull()) {
        ThrowReferenceError("Gobot node handle is invalid");
    }
    if (state->context != nullptr && !IsAppContextLive(state->context)) {
        ThrowReferenceError("Gobot node handle '" + state->name_snapshot +
                            "' belongs to a destroyed app context");
    }

    if (state->ownership != PyNodeOwnership::DetachedOwned &&
        state->scene_epoch != 0 &&
        state->context != nullptr &&
        state->scene_epoch != SceneEpochForContext(state->context)) {
        ThrowReferenceError("Gobot node handle '" + state->name_snapshot +
                            "' is from an inactive scene epoch");
    }

    Node* node = TryResolve();
    if (node == nullptr) {
        ThrowReferenceError("Gobot node handle '" + state->name_snapshot +
                            "' id=" + std::to_string(static_cast<std::uint64_t>(state->id)) +
                            " no longer resolves to a live node");
    }
    return node;
}

void PyNodeHandle::TransferToDetachedOwned(Node* node) {
    if (!state) {
        state = MakeState(node, ActiveSceneContext(), ActiveSceneEpoch(), PyNodeOwnership::DetachedOwned);
        return;
    }
    state->id = node != nullptr ? node->GetInstanceId() : ObjectID();
    state->context = ActiveSceneContext();
    state->scene_epoch = ActiveSceneEpoch();
    state->ownership = PyNodeOwnership::DetachedOwned;
    RefreshSnapshot(node);
}

std::string ExpectedTypeForNode(Node* node) {
    return node == nullptr ? "Node" : std::string(node->GetClassStringName());
}

EngineContext* ResolveHandleContext(EngineContext* context) {
    return context != nullptr ? context : ActiveSceneContext();
}

std::uint64_t ResolveHandleEpoch(EngineContext* context, std::uint64_t epoch) {
    if (epoch != 0) {
        return epoch;
    }
    return SceneEpochForContext(ResolveHandleContext(context));
}

PyNodeHandle MakeNodeHandle(Node* node,
                            PyNodeOwnership ownership,
                            EngineContext* context,
                            std::uint64_t epoch) {
    EngineContext* resolved_context = ResolveHandleContext(context);
    return PyNodeHandle(node, ExpectedTypeForNode(node), resolved_context, ResolveHandleEpoch(resolved_context, epoch), ownership);
}

PyNode3DHandle MakeNode3DHandle(Node3D* node,
                                PyNodeOwnership ownership,
                                EngineContext* context,
                                std::uint64_t epoch) {
    EngineContext* resolved_context = ResolveHandleContext(context);
    return PyNode3DHandle(node, ExpectedTypeForNode(node), resolved_context, ResolveHandleEpoch(resolved_context, epoch), ownership);
}

PyRobot3DHandle MakeRobot3DHandle(Robot3D* node,
                                  PyNodeOwnership ownership,
                                  EngineContext* context,
                                  std::uint64_t epoch) {
    EngineContext* resolved_context = ResolveHandleContext(context);
    return PyRobot3DHandle(node, "Robot3D", resolved_context, ResolveHandleEpoch(resolved_context, epoch), ownership);
}

PyLink3DHandle MakeLink3DHandle(Link3D* node,
                                PyNodeOwnership ownership,
                                EngineContext* context,
                                std::uint64_t epoch) {
    EngineContext* resolved_context = ResolveHandleContext(context);
    return PyLink3DHandle(node, "Link3D", resolved_context, ResolveHandleEpoch(resolved_context, epoch), ownership);
}

PyJoint3DHandle MakeJoint3DHandle(Joint3D* node,
                                  PyNodeOwnership ownership,
                                  EngineContext* context,
                                  std::uint64_t epoch) {
    EngineContext* resolved_context = ResolveHandleContext(context);
    return PyJoint3DHandle(node, "Joint3D", resolved_context, ResolveHandleEpoch(resolved_context, epoch), ownership);
}

PyCollisionShape3DHandle MakeCollisionShape3DHandle(CollisionShape3D* node,
                                                    PyNodeOwnership ownership,
                                                    EngineContext* context,
                                                    std::uint64_t epoch) {
    EngineContext* resolved_context = ResolveHandleContext(context);
    return PyCollisionShape3DHandle(node, "CollisionShape3D", resolved_context, ResolveHandleEpoch(resolved_context, epoch), ownership);
}

PyMeshInstance3DHandle MakeMeshInstance3DHandle(MeshInstance3D* node,
                                                PyNodeOwnership ownership,
                                                EngineContext* context,
                                                std::uint64_t epoch) {
    EngineContext* resolved_context = ResolveHandleContext(context);
    return PyMeshInstance3DHandle(node, "MeshInstance3D", resolved_context, ResolveHandleEpoch(resolved_context, epoch), ownership);
}

PyTerrain3DHandle MakeTerrain3DHandle(Terrain3D* node,
                                      PyNodeOwnership ownership,
                                      EngineContext* context,
                                      std::uint64_t epoch) {
    EngineContext* resolved_context = ResolveHandleContext(context);
    return PyTerrain3DHandle(node, "Terrain3D", resolved_context, ResolveHandleEpoch(resolved_context, epoch), ownership);
}

PySensor3DHandle MakeSensor3DHandle(Sensor3D* node,
                                    PyNodeOwnership ownership,
                                    EngineContext* context,
                                    std::uint64_t epoch) {
    EngineContext* resolved_context = ResolveHandleContext(context);
    return PySensor3DHandle(node, "Sensor3D", resolved_context, ResolveHandleEpoch(resolved_context, epoch), ownership);
}

PyIMUSensor3DHandle MakeIMUSensor3DHandle(IMUSensor3D* node,
                                          PyNodeOwnership ownership,
                                          EngineContext* context,
                                          std::uint64_t epoch) {
    EngineContext* resolved_context = ResolveHandleContext(context);
    return PyIMUSensor3DHandle(node, "IMUSensor3D", resolved_context, ResolveHandleEpoch(resolved_context, epoch), ownership);
}

PyAngularMomentumSensor3DHandle MakeAngularMomentumSensor3DHandle(AngularMomentumSensor3D* node,
                                                                  PyNodeOwnership ownership,
                                                                  EngineContext* context,
                                                                  std::uint64_t epoch) {
    EngineContext* resolved_context = ResolveHandleContext(context);
    return PyAngularMomentumSensor3DHandle(
            node,
            "AngularMomentumSensor3D",
            resolved_context,
            ResolveHandleEpoch(resolved_context, epoch),
            ownership);
}

PyContactSensor3DHandle MakeContactSensor3DHandle(ContactSensor3D* node,
                                                  PyNodeOwnership ownership,
                                                  EngineContext* context,
                                                  std::uint64_t epoch) {
    EngineContext* resolved_context = ResolveHandleContext(context);
    return PyContactSensor3DHandle(node, "ContactSensor3D", resolved_context, ResolveHandleEpoch(resolved_context, epoch), ownership);
}

PyRayCastSensor3DHandle MakeRayCastSensor3DHandle(RayCastSensor3D* node,
                                                  PyNodeOwnership ownership,
                                                  EngineContext* context,
                                                  std::uint64_t epoch) {
    EngineContext* resolved_context = ResolveHandleContext(context);
    return PyRayCastSensor3DHandle(node, "RayCastSensor3D", resolved_context, ResolveHandleEpoch(resolved_context, epoch), ownership);
}

PyTerrainHeightSensor3DHandle MakeTerrainHeightSensor3DHandle(
        TerrainHeightSensor3D* node,
        PyNodeOwnership ownership,
        EngineContext* context,
        std::uint64_t epoch) {
    EngineContext* resolved_context = ResolveHandleContext(context);
    return PyTerrainHeightSensor3DHandle(node, "TerrainHeightSensor3D", resolved_context, ResolveHandleEpoch(resolved_context, epoch), ownership);
}

PyHeightScanner3DHandle MakeHeightScanner3DHandle(
        HeightScanner3D* node,
        PyNodeOwnership ownership,
        EngineContext* context,
        std::uint64_t epoch) {
    EngineContext* resolved_context = ResolveHandleContext(context);
    return PyHeightScanner3DHandle(node, "HeightScanner3D", resolved_context, ResolveHandleEpoch(resolved_context, epoch), ownership);
}

PyVelocityCommandDebug3DHandle MakeVelocityCommandDebug3DHandle(
        VelocityCommandDebug3D* node,
        PyNodeOwnership ownership,
        EngineContext* context,
        std::uint64_t epoch) {
    EngineContext* resolved_context = ResolveHandleContext(context);
    return PyVelocityCommandDebug3DHandle(
            node,
            "VelocityCommandDebug3D",
            resolved_context,
            ResolveHandleEpoch(resolved_context, epoch),
            ownership);
}

PyNodeHandle MakeTypedNodeHandle(Node* node,
                                 PyNodeOwnership ownership,
                                 EngineContext* context,
                                 std::uint64_t epoch) {
    if (auto* velocity_debug = Object::PointerCastTo<VelocityCommandDebug3D>(node)) {
        return MakeVelocityCommandDebug3DHandle(velocity_debug, ownership, context, epoch);
    }
    if (auto* mesh_instance = Object::PointerCastTo<MeshInstance3D>(node)) {
        return MakeMeshInstance3DHandle(mesh_instance, ownership, context, epoch);
    }
    if (auto* terrain = Object::PointerCastTo<Terrain3D>(node)) {
        return MakeTerrain3DHandle(terrain, ownership, context, epoch);
    }
    if (auto* contact_sensor = Object::PointerCastTo<ContactSensor3D>(node)) {
        return MakeContactSensor3DHandle(contact_sensor, ownership, context, epoch);
    }
    if (auto* height_scanner = Object::PointerCastTo<HeightScanner3D>(node)) {
        return MakeHeightScanner3DHandle(height_scanner, ownership, context, epoch);
    }
    if (auto* terrain_height_sensor = Object::PointerCastTo<TerrainHeightSensor3D>(node)) {
        return MakeTerrainHeightSensor3DHandle(terrain_height_sensor, ownership, context, epoch);
    }
    if (auto* raycast_sensor = Object::PointerCastTo<RayCastSensor3D>(node)) {
        return MakeRayCastSensor3DHandle(raycast_sensor, ownership, context, epoch);
    }
    if (auto* angular_momentum_sensor = Object::PointerCastTo<AngularMomentumSensor3D>(node)) {
        return MakeAngularMomentumSensor3DHandle(angular_momentum_sensor, ownership, context, epoch);
    }
    if (auto* imu_sensor = Object::PointerCastTo<IMUSensor3D>(node)) {
        return MakeIMUSensor3DHandle(imu_sensor, ownership, context, epoch);
    }
    if (auto* sensor = Object::PointerCastTo<Sensor3D>(node)) {
        return MakeSensor3DHandle(sensor, ownership, context, epoch);
    }
    if (auto* collision_shape = Object::PointerCastTo<CollisionShape3D>(node)) {
        return MakeCollisionShape3DHandle(collision_shape, ownership, context, epoch);
    }
    if (auto* joint = Object::PointerCastTo<Joint3D>(node)) {
        return MakeJoint3DHandle(joint, ownership, context, epoch);
    }
    if (auto* link = Object::PointerCastTo<Link3D>(node)) {
        return MakeLink3DHandle(link, ownership, context, epoch);
    }
    if (auto* robot = Object::PointerCastTo<Robot3D>(node)) {
        return MakeRobot3DHandle(robot, ownership, context, epoch);
    }
    if (auto* node_3d = Object::PointerCastTo<Node3D>(node)) {
        return MakeNode3DHandle(node_3d, ownership, context, epoch);
    }
    return MakeNodeHandle(node, ownership, context, epoch);
}

py::object MakeTypedNodeObject(Node* node,
                               PyNodeOwnership ownership,
                               EngineContext* context,
                               std::uint64_t epoch) {
    if (auto* velocity_debug = Object::PointerCastTo<VelocityCommandDebug3D>(node)) {
        return py::cast(MakeVelocityCommandDebug3DHandle(velocity_debug, ownership, context, epoch));
    }
    if (auto* mesh_instance = Object::PointerCastTo<MeshInstance3D>(node)) {
        return py::cast(MakeMeshInstance3DHandle(mesh_instance, ownership, context, epoch));
    }
    if (auto* terrain = Object::PointerCastTo<Terrain3D>(node)) {
        return py::cast(MakeTerrain3DHandle(terrain, ownership, context, epoch));
    }
    if (auto* contact_sensor = Object::PointerCastTo<ContactSensor3D>(node)) {
        return py::cast(MakeContactSensor3DHandle(contact_sensor, ownership, context, epoch));
    }
    if (auto* height_scanner = Object::PointerCastTo<HeightScanner3D>(node)) {
        return py::cast(MakeHeightScanner3DHandle(height_scanner, ownership, context, epoch));
    }
    if (auto* terrain_height_sensor = Object::PointerCastTo<TerrainHeightSensor3D>(node)) {
        return py::cast(MakeTerrainHeightSensor3DHandle(terrain_height_sensor, ownership, context, epoch));
    }
    if (auto* raycast_sensor = Object::PointerCastTo<RayCastSensor3D>(node)) {
        return py::cast(MakeRayCastSensor3DHandle(raycast_sensor, ownership, context, epoch));
    }
    if (auto* angular_momentum_sensor = Object::PointerCastTo<AngularMomentumSensor3D>(node)) {
        return py::cast(MakeAngularMomentumSensor3DHandle(angular_momentum_sensor, ownership, context, epoch));
    }
    if (auto* imu_sensor = Object::PointerCastTo<IMUSensor3D>(node)) {
        return py::cast(MakeIMUSensor3DHandle(imu_sensor, ownership, context, epoch));
    }
    if (auto* sensor = Object::PointerCastTo<Sensor3D>(node)) {
        return py::cast(MakeSensor3DHandle(sensor, ownership, context, epoch));
    }
    if (auto* collision_shape = Object::PointerCastTo<CollisionShape3D>(node)) {
        return py::cast(MakeCollisionShape3DHandle(collision_shape, ownership, context, epoch));
    }
    if (auto* joint = Object::PointerCastTo<Joint3D>(node)) {
        return py::cast(MakeJoint3DHandle(joint, ownership, context, epoch));
    }
    if (auto* link = Object::PointerCastTo<Link3D>(node)) {
        return py::cast(MakeLink3DHandle(link, ownership, context, epoch));
    }
    if (auto* robot = Object::PointerCastTo<Robot3D>(node)) {
        return py::cast(MakeRobot3DHandle(robot, ownership, context, epoch));
    }
    if (auto* node_3d = Object::PointerCastTo<Node3D>(node)) {
        return py::cast(MakeNode3DHandle(node_3d, ownership, context, epoch));
    }
    return py::cast(MakeNodeHandle(node, ownership, context, epoch));
}

void ExecuteSetNodeProperty(Node* node, const std::string& property_name, Variant value) {
    if (node == nullptr) {
        throw std::invalid_argument("cannot set property on a null Gobot node");
    }
    ScopedActiveAppContext scoped_context(ContextForNode(node));
    if (IsSceneScriptRuntimeMutation()) {
        if (!node->Set(property_name, std::move(value))) {
            throw std::runtime_error("failed to set Gobot runtime node property '" + property_name + "'");
        }
        return;
    }
    ExecuteSceneMutation("set_property", [&]() {
        return std::make_unique<SetNodePropertyCommand>(node->GetInstanceId(), property_name, std::move(value));
    });
}

EngineContext* ContextForNode(const Node* node) {
    if (EngineContext* context = FindAppContextForSceneRoot(node)) {
        return context;
    }
    return GetActiveAppContextOrNull();
}

} // namespace gobot::python
