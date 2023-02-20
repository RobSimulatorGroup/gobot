/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-19
*/

#include <gobot/log.hpp>
#include <LLGL/LLGL.h>
#include <LLGL/Misc/TypeNames.h>
#include <LLGL/Misc/VertexFormat.h>
#include <LLGL/Misc/Utility.h>

int main()  {
    std::unique_ptr<LLGL::RenderSystem> renderer = LLGL::RenderSystem::Load("OpenGL");

    // Create swap-chain
    LLGL::SwapChainDescriptor swapChainDesc;
    {
        swapChainDesc.resolution    = { 800, 600 };
        swapChainDesc.depthBits     = 0; // We don't need a depth buffer for this example
        swapChainDesc.stencilBits   = 0; // We don't need a stencil buffer for this example
        swapChainDesc.samples       = 8; // check if LLGL adapts sample count that is too high
    }
    LLGL::SwapChain* swapChain = renderer->CreateSwapChain(swapChainDesc);

    // Print renderer information
    const auto& info = renderer->GetRendererInfo();

    std::cout << "Renderer:             " << info.rendererName << std::endl;
    std::cout << "Device:               " << info.deviceName << std::endl;
    std::cout << "Vendor:               " << info.vendorName << std::endl;
    std::cout << "Shading Language:     " << info.shadingLanguageName << std::endl;
    std::cout << "Swap Chain Format:    " << LLGL::ToString(swapChain->GetColorFormat()) << std::endl;
    std::cout << "Depth/Stencil Format: " << LLGL::ToString(swapChain->GetDepthStencilFormat()) << std::endl;

    // Enable V-sync
    swapChain->SetVsyncInterval(1);

    // Set window title and show window
    auto& window = LLGL::CastTo<LLGL::Window>(swapChain->GetSurface());

    window.SetTitle(L"LLGL Example: Hello Triangle");
    window.Show();

    // Vertex data structure
    struct Vertex
    {
        float   position[2];
        uint8_t color[4];
    };

    // Vertex data (3 vertices for our triangle)
    const float s = 0.5f;

    Vertex vertices[] =
            {
                    { {  0,  s }, { 255, 0, 0, 255 } }, // 1st vertex: center-top, red
                    { {  s, -s }, { 0, 255, 0, 255 } }, // 2nd vertex: right-bottom, green
                    { { -s, -s }, { 0, 0, 255, 255 } }, // 3rd vertex: left-bottom, blue
            };

    // Vertex format
    LLGL::VertexFormat vertexFormat;

    // Append 2D float vector for position attribute
    vertexFormat.AppendAttribute({ "position", LLGL::Format::RG32Float });

    // Append 3D unsigned byte vector for color
    vertexFormat.AppendAttribute({ "color",    LLGL::Format::RGBA8UNorm });

    // Update stride in case out vertex structure is not 4-byte aligned
    vertexFormat.SetStride(sizeof(Vertex));

    // Create vertex buffer
    LLGL::BufferDescriptor vertexBufferDesc;
    {
        vertexBufferDesc.size           = sizeof(vertices);                 // Size (in bytes) of the vertex buffer
        vertexBufferDesc.bindFlags      = LLGL::BindFlags::VertexBuffer;    // Enables the buffer to be bound to a vertex buffer slot
        vertexBufferDesc.vertexAttribs  = vertexFormat.attributes;          // Vertex format layout
    }
    LLGL::Buffer* vertexBuffer = renderer->CreateBuffer(vertexBufferDesc, vertices);

    // Create shaders
    LLGL::Shader* vertShader = nullptr;
    LLGL::Shader* fragShader = nullptr;

    const auto& languages = renderer->GetRenderingCaps().shadingLanguages;

    LLGL::ShaderDescriptor vertShaderDesc, fragShaderDesc;

    vertShaderDesc = LLGL::ShaderDescFromFile(LLGL::ShaderType::Vertex,   "Example.450core.vert");
    fragShaderDesc = LLGL::ShaderDescFromFile(LLGL::ShaderType::Fragment, "Example.450core.frag");

    // Specify vertex attributes for vertex shader
    vertShaderDesc.vertex.inputAttribs = vertexFormat.attributes;

    vertShader = renderer->CreateShader(vertShaderDesc);
    fragShader = renderer->CreateShader(fragShaderDesc);

    for (auto shader : { vertShader, fragShader })
    {
        if (auto report = shader->GetReport())
            std::cerr << report->GetText() << std::endl;
    }

    // Create graphics pipeline
    LLGL::PipelineState* pipeline = nullptr;
    std::unique_ptr<LLGL::Blob> pipelineCache;

    {
        LLGL::GraphicsPipelineDescriptor pipelineDesc;
        {
            pipelineDesc.vertexShader                   = vertShader;
            pipelineDesc.fragmentShader                 = fragShader;
            pipelineDesc.renderPass                     = swapChain->GetRenderPass();
            pipelineDesc.rasterizer.multiSampleEnabled  = (swapChainDesc.samples > 1);
        }


        // Create graphics PSO
        pipeline = renderer->CreatePipelineState(pipelineDesc);


        // Link shader program and check for errors
        if (auto report = pipeline->GetReport())
        {
            if (report->HasErrors())
                throw std::runtime_error(report->GetText());
        }
    }

    // Create command buffer to submit subsequent graphics commands to the GPU
    LLGL::CommandBuffer* commands = renderer->CreateCommandBuffer(LLGL::CommandBufferFlags::ImmediateSubmit);

    // Enter main loop
    while (window.PollEvents())
    {

//        window.SetSize(LLGL::Extent2D(1000, 800));
        // Begin recording commands
        commands->Begin();
        {
            // Set viewport and scissor rectangle
            commands->SetViewport(swapChain->GetResolution());

            // Set graphics pipeline
            commands->SetPipelineState(*pipeline);

            // Set vertex buffer
            commands->SetVertexBuffer(*vertexBuffer);

            // Set the swap-chain as the initial render target
            commands->BeginRenderPass(*swapChain);
            {
                // Clear color buffer
                commands->Clear(LLGL::ClearFlags::Color);

                // Draw triangle with 3 vertices
                commands->Draw(3, 0);
            }
            commands->EndRenderPass();
        }
        commands->End();

        // Present the result on the screen
        swapChain->Present();
    }


}