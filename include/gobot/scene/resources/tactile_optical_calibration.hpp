#pragma once
#include "gobot/core/io/resource.hpp"
#include "gobot/core/tactile_optical_data.hpp"
namespace gobot {
class GOBOT_EXPORT TactileOpticalCalibration : public Resource {
    GOBCLASS(TactileOpticalCalibration, Resource)
  public:
    void SetWidth(std::uint32_t value);
    std::uint32_t GetWidth() const;
    void SetHeight(std::uint32_t value);
    std::uint32_t GetHeight() const;
    void SetBins(std::uint32_t value);
    std::uint32_t GetBins() const;
    void SetCalibrationWidth(std::uint32_t value);
    std::uint32_t GetCalibrationWidth() const;
    void SetCalibrationHeight(std::uint32_t value);
    std::uint32_t GetCalibrationHeight() const;
    void SetCoefficients(const std::vector<RealType> &value);
    const std::vector<RealType> &GetCoefficients() const;
    void SetBackground(const std::vector<RealType> &value);
    const std::vector<RealType> &GetBackground() const;
    void SetPixelSize(RealType value);
    RealType GetPixelSize() const;
    const TactileOpticalData &GetData() const { return data_; }

  private:
    TactileOpticalData data_;
};
} // namespace gobot
