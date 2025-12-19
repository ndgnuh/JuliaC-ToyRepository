module PredatorPreyGrass

import TOML

using Random
using LinearAlgebra
using Raylib
using UnPack: @unpack
using StaticArrays: SVector
using FixedPointNumbers: N0f8
using Base.Threads: @threads

import Raylib.Binding as raylib
import
    ..DisplayConfig,
    ..stdout,
    ..stderr

const Color = Raylib.RayColor

@kwdef mutable struct Animals
    "A [2, n] matrix of positions of all Wolves"
    positions::Matrix{Float64} = []

    "A [2, n] matrix of velocities of all Wolves"
    velocities::Matrix{Float64} = []

    "A vector of energies of all Wolves"
    energies::Vector{Float64} = []
end


@kwdef struct ModelConfig
    n_predators::Int = 50
    n_preys::Int = 500
    dims::Tuple{Int, Int} = (100, 100)
    regrowth_time::Int = 100
    momentum::Float64 = 0.8
    δenergy_prey::Float64 = 0.4
    δenergy_predator::Float64 = 0.2
    prey_speed::Float64
    predator_speed::Float64
    prey_reproduce::Float64 = 0.04
    predator_reproduce::Float64 = 0.05
    hunt_distance::Float64 = 10.0
    seed::Int = 0
end


function config_from_toml(path::AbstractString)::ModelConfig
    cfg = TOML.parsefile(path)
    dims = cfg["dims"]::Vector{Int}
    n_dims = length(dims)

    @assert n_dims == 2 "Invalid dimension size $n_dims"
    @assert all(dims .> 0) "Dimensions must be positive"
    keys_types = [
        "n_predators" => Int,
        "n_preys" => Int,
        "regrowth_time" => Int,
        "momentum" => Number,
        "delta_energy_prey" => Number,
        "delta_energy_predator" => Number,
        "prey_reproduce" => Number,
        "predator_reproduce" => Number,
        "prey_speed" => Number,
        "predator_speed" => Number,
        "hunt_distance" => Number,
    ]
    for (key, type) in keys_types
        @assert haskey(cfg, key) "Missing key '$key' in config file"
        @assert isa(cfg[key], type) "Invalid type for key '$key'"
    end

    config = ModelConfig(;
        n_predators = cfg["n_predators"]::Int,
        n_preys = cfg["n_preys"]::Int,
        dims = (dims[1], dims[2]),
        regrowth_time = cfg["regrowth_time"]::Int,
        momentum = cfg["momentum"]::Float64,
        δenergy_prey = cfg["delta_energy_prey"]::Float64,
        δenergy_predator = cfg["delta_energy_predator"]::Float64,
        prey_reproduce = cfg["prey_reproduce"]::Float64,
        hunt_distance = cfg["hunt_distance"]::Float64,
        predator_reproduce = cfg["predator_reproduce"]::Float64,
        prey_speed = cfg["prey_speed"]::Float64,
        predator_speed = cfg["predator_speed"]::Float64,
        seed = get(cfg, "seed", 0)::Int,
    )

    return config
end

@kwdef mutable struct Model{RNG <: AbstractRNG}
    step::Int = 0
    const config::ModelConfig
    const preys::Animals
    const predators::Animals
    const grass::Matrix{Bool}
    const grass_countdown::Matrix{Int}
    const rng::RNG = Xoshiro(config.seed)
end

function init_model(config::ModelConfig)::Model
    dims::Tuple{Int, Int} = config.dims::Tuple{Int, Int}
    w, h = dims

    rng = Xoshiro(config.seed)

    grass = rand(rng, (false, true), w, h)
    grass_countdown = rand(rng, 1:config.regrowth_time, w, h)


    function generate_animals(n::Int, δenergy::AbstractFloat)::Animals
        xs = rand(rng, range(1, w, w * 10), n)
        ys = rand(rng, range(1, h, h * 10), n)
        positions = vcat(xs', ys')
        velocities = randn(Float64, 2, n)
        energies = rand(1:0.01:(2 * δenergy), n)
        return Animals(
            positions = positions,
            velocities = velocities,
            energies = energies,
        )
    end

    preys = generate_animals(config.n_preys, config.δenergy_prey)::Animals
    predators = generate_animals(config.n_predators, config.δenergy_predator)::Animals

    return Model(; rng, grass, grass_countdown, predators, preys, config)
end

function draw_heatmap!(config::DisplayConfig, data::Matrix{Bool})::Nothing
    width = Raylib.Binding.GetScreenWidth()
    height = Raylib.Binding.GetScreenHeight()

    border_size_x = 10
    border_size_y = 100

    canvas_x = border_size_x + 1
    canvas_y = border_size_y + 1
    canvas_width = width - border_size_x * 2
    canvas_height = height - border_size_y * 2

    raylib.ClearBackground(Raylib.BLACK)
    raylib.DrawRectangle(
        canvas_x, canvas_y,
        canvas_width, canvas_height,
        Raylib.RAYWHITE
    )


    rows, cols = size(data)
    cell_width = (canvas_width / cols)::Float64
    cell_height = (canvas_height / rows)::Float64
    cell_size = SVector{2, Float64}(cell_width, cell_height)

    for y::Int in 1:rows, x::Int in 1:cols

        cell_x::Float64 = (x - 1) * cell_width + canvas_x
        cell_y::Float64 = (y - 1) * cell_height + canvas_y
        position = SVector{2, Float64}(cell_x, cell_y)

        value = data[y, x]::Bool
        color = value ? Raylib.GREEN : Raylib.WHITE

        raylib.DrawRectangleV(
            position, cell_size,
            color,
        )
    end

    return nothing
end

function draw_animals!(
        config::DisplayConfig, model::Model, animals::Animals, color::Color
    )::Nothing
    width = Raylib.Binding.GetScreenWidth()
    height = Raylib.Binding.GetScreenHeight()

    border_size = 50

    @unpack config = model
    map_h, map_w = model.config.dims

    border_size_x = 10
    border_size_y = 100

    canvas_x = border_size_x + 1
    canvas_y = border_size_y + 1
    canvas_width = width - border_size_x * 2
    canvas_height = height - border_size_y * 2
    n_animals = size(animals.positions, 2)


    for i::Int in 1:n_animals
        if animals.energies[i] <= 0.0
            continue
        end

        x_pos = animals.positions[1, i]::Float64
        y_pos = animals.positions[2, i]::Float64
        x_pos = x_pos / map_w * canvas_width + canvas_x
        y_pos = y_pos / map_h * canvas_height + canvas_y


        raylib.DrawCircleV(
            SVector(x_pos, y_pos),
            5.0,
            color,
        )
    end

    return nothing
end

function visualize(model::Model, config::DisplayConfig)::Nothing
    raylib.ClearBackground(Raylib.RAYWHITE)


    width = Raylib.Binding.GetScreenWidth()
    draw_heatmap!(config, model.grass)
    draw_animals!(config, model, model.preys, Raylib.BLUE)
    draw_animals!(config, model, model.predators, Raylib.RED)

    raylib.DrawFPS(width - 100, 5)
    raylib.DrawText(
        "Step $(model.step)",
        5, 5, 20, Raylib.WHITE
    )
    raylib.DrawText(
        "Preys $(model.preys.energies |> sum)",
        5, 25, 20, Raylib.WHITE
    )
    raylib.DrawText(
        "Predators $(model.predators.energies |> sum)",
        5, 45, 20, Raylib.WHITE
    )

    return nothing
end

const angles = range(-π / 3, π / 3, 100000)

function random_velocity(
        rng::AbstractRNG,
        x::AbstractFloat, y::AbstractFloat,
        vx::AbstractFloat, vy::AbstractFloat,
        momentum::AbstractFloat,
        speed::AbstractFloat, map_w::Int, map_h::Int,
    )::Tuple{AbstractFloat, AbstractFloat}

    # Rotate velocity randomly
    ϕ = rand(rng, angles)::AbstractFloat
    cosϕ = cos(ϕ)
    sinϕ = sin(ϕ)
    vx, vy = let
        # Rotate the velocity vector by angle ϕ with some momentum
        # Apply some noise as well
        vx_ = momentum * vx + (1 - momentum) * (vx * cosϕ - vy * sinϕ)
        vy_ = momentum * vy + (1 - momentum) * (vx * sinϕ + vy * cosϕ)
        (vx_, vy_)
    end

    # Normalize velocity
    norm = sqrt(vx^2 + vy^2)
    vx = vx / norm * speed
    vy = vy / norm * speed

    # Bounce back from walls
    if x + vx < 0 || x + vx > map_w
        vx = -vx
    end
    if y + vy < 0 || y + vy > map_h
        vy = -vy
    end

    return vx, vy
end

function reproduce!(animals::Animals, i::Int)::Nothing
    new_energy = animals.energies[i] / 2
    new_position = animals.positions[:, i]
    new_velocity = animals.velocities[:, i]
    animals.energies[i] = new_energy

    # If a dead one exists, reuse its slot
    # ====================================
    j = findfirst(<=(0.0), animals.energies)
    if isnothing(j)
        push!(animals.energies, new_energy)
        animals.positions = hcat(animals.positions, new_position)
        animals.velocities = hcat(animals.velocities, new_velocity)
    else
        animals.positions[:, j] .= new_position
        animals.velocities[:, j] = new_velocity
        animals.energies[j] = new_energy
    end
    return nothing
end


function step!(model::Model)::Nothing
    @unpack (
        rng, config,
        preys, predators,
        grass, grass_countdown,
    ) = model
    @unpack (
        momentum, prey_speed, predator_speed,
        δenergy_prey, δenergy_predator, regrowth_time,
        hunt_distance, dims,
        predator_reproduce, prey_reproduce,
    ) = config

    map_w, map_h = dims

    model.step += 1

    # Regrow grass logistically
    for c in 1:map_w, r in 1:map_h
        if grass_countdown[r, c] > 0
            grass_countdown[r, c] -= 1
        else
            grass[r, c] = true
            grass_countdown[r, c] = regrowth_time
        end
    end

    num_preys = size(preys.positions, 2)
    num_predators = size(predators.positions, 2)

    for i in 1:num_preys
        energy_prey = preys.energies[i]
        if energy_prey <= 0.0
            continue
        end

        # Consume energy
        # ==============
        preys.energies[i] -= 1

        # Move prey
        # =========
        vx, vy = random_velocity(
            rng,
            preys.positions[1, i], preys.positions[2, i],
            preys.velocities[1, i], preys.velocities[2, i],
            momentum, prey_speed, map_w, map_h,
        )
        preys.velocities[1, i] = vx
        preys.velocities[2, i] = vy
        preys.positions[1, i] += vx
        preys.positions[2, i] += vy

        # Eat grass
        # =========
        x, y = round.(Int, preys.positions[:, i])
        x = clamp(x, 1, map_w)
        y = clamp(y, 1, map_h)
        if grass[y, x]
            preys.energies[i] += δenergy_prey
            grass[y, x] = false
        end

        #= print("Vx: $vx, Vy: $vy\n") =#

        # Reproduce
        # =========
        if rand(rng) <= config.prey_reproduce
            reproduce!(preys, i)
        end
    end

    for i in 1:num_predators
        # Skip dead predators
        if predators.energies[i] <= 0.0
            continue
        end

        # Consume energy
        # ==============
        predators.energies[i] -= 1

        # Move predator randomly
        # ======================
        vx, vy = random_velocity(
            rng,
            predators.positions[1, i], predators.positions[2, i],
            predators.velocities[1, i], predators.velocities[2, i],
            momentum, predator_speed, map_w, map_h,
        )
        predators.velocities[1, i] = vx
        predators.velocities[2, i] = vy
        predators.positions[1, i] += vx
        predators.positions[2, i] += vy

        # Eat prey
        # =========
        for j in 1:num_preys
            energy_prey = preys.energies[j]
            if energy_prey <= 0.0
                continue
            end

            dx = predators.positions[1, i] - preys.positions[1, j]
            dy = predators.positions[2, i] - preys.positions[2, j]
            distance::Float64 = sqrt(dx^2 + dy^2)
            if distance <= hunt_distance
                # Eat prey
                predators.energies[i] += δenergy_predator
                preys.energies[j] = 0
                @goto break_inner
            end
        end
        @label break_inner

        # Reproduce
        # =========
        if rand(rng) <= predator_reproduce
            reproduce!(predators, i)
        end
    end

    # Placeholder for model update logic
    return nothing
end


function Base.print(io::IO, config::ModelConfig)::Nothing
    println(io, "ModelConfig:")
    for (field) in fieldnames(ModelConfig)
        println(io, "\t$(field) = $(getfield(config, field))")
    end
    return nothing
end


end
