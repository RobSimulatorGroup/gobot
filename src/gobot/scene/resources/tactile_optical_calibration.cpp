#include "gobot/scene/resources/tactile_optical_calibration.hpp"
#include "gobot/core/registration.hpp"
namespace gobot {
void TactileOpticalCalibration::SetPixelSize(RealType value) {
    data_.pixel_size = value;
    MarkChanged();
}
RealType TactileOpticalCalibration::GetPixelSize() const { return data_.pixel_size; }
void TactileOpticalCalibration::SetWidth(std::uint32_t v) {
    data_.width = v;
    MarkChanged();
}
std::uint32_t TactileOpticalCalibration::GetWidth() const { return data_.width; }
void TactileOpticalCalibration::SetHeight(std::uint32_t v) {
    data_.height = v;
    MarkChanged();
}
std::uint32_t TactileOpticalCalibration::GetHeight() const { return data_.height; }
void TactileOpticalCalibration::SetBins(std::uint32_t v) {
    data_.bins = v;
    MarkChanged();
}
std::uint32_t TactileOpticalCalibration::GetBins() const { return data_.bins; }
void TactileOpticalCalibration::SetCalibrationWidth(std::uint32_t v) {
    data_.calibration_width = v;
    MarkChanged();
}
std::uint32_t TactileOpticalCalibration::GetCalibrationWidth() const { return data_.calibration_width; }
void TactileOpticalCalibration::SetCalibrationHeight(std::uint32_t v) {
    data_.calibration_height = v;
    MarkChanged();
}
std::uint32_t TactileOpticalCalibration::GetCalibrationHeight() const { return data_.calibration_height; }
void TactileOpticalCalibration::SetCoefficients(const std::vector<RealType> &v) {
    data_.coefficients = v;
    MarkChanged();
}
const std::vector<RealType> &TactileOpticalCalibration::GetCoefficients() const { return data_.coefficients; }
void TactileOpticalCalibration::SetBackground(const std::vector<RealType> &v) {
    data_.background = v;
    MarkChanged();
}
const std::vector<RealType> &TactileOpticalCalibration::GetBackground() const { return data_.background; }
} // namespace gobot
GOBOT_REGISTRATION {
    Class_<gobot::TactileOpticalCalibration>("TactileOpticalCalibration")
        .constructor()(CtorAsRawPtr)
        .property("pixel_size", &gobot::TactileOpticalCalibration::GetPixelSize,
                  &gobot::TactileOpticalCalibration::SetPixelSize)
        .property("width", &gobot::TactileOpticalCalibration::GetWidth,
                  &gobot::TactileOpticalCalibration::SetWidth)
        .property("height", &gobot::TactileOpticalCalibration::GetHeight,
                  &gobot::TactileOpticalCalibration::SetHeight)
        .property("bins", &gobot::TactileOpticalCalibration::GetBins,
                  &gobot::TactileOpticalCalibration::SetBins)
        .property("calibration_width", &gobot::TactileOpticalCalibration::GetCalibrationWidth,
                  &gobot::TactileOpticalCalibration::SetCalibrationWidth)
        .property("calibration_height", &gobot::TactileOpticalCalibration::GetCalibrationHeight,
                  &gobot::TactileOpticalCalibration::SetCalibrationHeight)
        .property("coefficients", &gobot::TactileOpticalCalibration::GetCoefficients,
                  &gobot::TactileOpticalCalibration::SetCoefficients)
        .property("background", &gobot::TactileOpticalCalibration::GetBackground,
                  &gobot::TactileOpticalCalibration::SetBackground);
    gobot::Type::register_wrapper_converter_for_base_classes<gobot::Ref<gobot::TactileOpticalCalibration>,
                                                             gobot::Ref<gobot::Resource>>();
}
