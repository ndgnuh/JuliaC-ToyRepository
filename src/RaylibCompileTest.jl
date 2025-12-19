module RaylibCompileTest

include("monkey_patch.jl")
include("display.jl")

using Raylib
using TOML
using Accessors: @set

const raylib = Raylib.Binding
const stdout = Core.stdout
const stderr = Core.stderr

include("PredatorPreyGrass.jl")
using .PredatorPreyGrass:
    ModelConfig, init_model, visualize, step!,
    config_from_toml


using Random

function (@main)(ARGS)::Cint
    if length(ARGS) > 0
        input_file = first(ARGS)
    else
        input_file = "./config.toml"
    end

    # Read model config
    # =================
    model_config = config_from_toml(input_file)
    println("Model configuration")
    println("===================")
    println(model_config)
    model = init_model(model_config)

    # Read display config
    # ===================
    cfg = TOML.parsefile(input_file)
    _ = haskey(cfg, "display_width")
    @assert haskey(cfg, "display_width") "Missing config key 'display_width'"
    @assert haskey(cfg, "display_height") "Missing config key 'display_height'"
    display_config = DisplayConfig(
        width = cfg["display_width"]::Int,
        height = cfg["display_height"]::Int
    )

    # Initialize display
    # ==================
    raylib.InitWindow(
        display_config.width,
        display_config.height,
        "Predator-Prey-Grass Model"
    )
    raylib.SetConfigFlags(Raylib.Binding.FLAG_WINDOW_RESIZABLE |> Int)

    # Display loop
    # ============
    step_every = 0.01
    total_time_offset = 0.0
    while !raylib.WindowShouldClose()
        raylib.BeginDrawing()
        total_time = raylib.GetTime()


        # Re-initialize model, randomize seed
        if Raylib.Binding.IsKeyPressed(Raylib.Binding.KEY_R)
            model_config = @set model_config.seed = rand(Int)
            print("Reset the model with seed=$(model_config.seed)\n")
            model = init_model(model_config)
            total_time_offset = total_time
        end

        # Run faster/slower
        if Raylib.Binding.IsKeyPressed(Raylib.Binding.KEY_A)
            step_every /= 2
        end
        if Raylib.Binding.IsKeyPressed(Raylib.Binding.KEY_D)
            step_every *= 2
        end

        if (total_time - total_time_offset) / step_every > model.step
            step!(model)
        end

        visualize(model, display_config)
        raylib.EndDrawing()
    end

    return 0
end

function (_main)(ARGS)::Cint

    config::DisplayConfig = let config_file = "./config.toml"
        config_data = TOML.parsefile(config_file)

        # We HAVE TO add type guards here so that
        # the compiler knows what to expect.
        DisplayConfig(
            width = get(config_data, "width", 800)::Int,
            height = get(config_data, "height", 600)::Int,
        )
    end

    print(stdout, "Intializing ABM model\n\n")
    model_config = ModelConfig()::ModelConfig
    print(stdout, model_config)
    print(stdout, "\n")

    model = init_model(model_config)


    raylib.InitWindow(
        config.width, config.height, "Test"
    )

    while !raylib.WindowShouldClose()
        raylib.BeginDrawing()
        visualize(model, config)
        step!(model)
        raylib.EndDrawing()
    end
    return 0
end

end # module RaylibCompileTest
