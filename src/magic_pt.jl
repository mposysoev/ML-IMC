using ..ML_IMC

struct MagicPreComputedInput
    nn_params::NeuralNetParameters
    system_params::SystemParameters
    reference_rdf::Vector{Float64}
end

struct MagicReferenceData
    distance_matrices::Vector{Matrix{Float64}}
    precomputed_energy::Vector{Float64}
    g2_matrices::Vector{Matrix{Float64}}
end

struct PotentialData
    r_values::Vector{Float64}
    potential_values::Vector{Float64}
end

struct PotentialLookup
    lookup_r::Vector{Float64}
    lookup_v::Vector{Float64}
    r_step::Float64
    r_min::Float64
    r_max::Float64
end

struct CachedTrajectory
    frames::Vector{Frame}
    n_frames::Int
    boxes::Vector{Vector{Float64}}
end

function load_potential_data(filename::AbstractString)::PotentialData
    check_file(filename)
    r_values = Float64[]
    potential_values = Float64[]

    open(filename, "r") do file
        for line in eachline(file)
            stripped_line = strip(line)
            (isempty(stripped_line) || startswith(stripped_line, "#")) && continue
            parts = split(stripped_line)
            if length(parts) >= 2
                try
                    r = parse(Float64, parts[1])
                    v = parse(Float64, parts[2])
                    push!(r_values, r)
                    push!(potential_values, v)
                catch e
                    if e isa ArgumentError
                        @warn "Skipping invalid numeric data in line: $line"
                    else
                        rethrow()
                    end
                end
            else
                @warn "Skipping line with insufficient columns: $line"
            end
        end
    end

    if isempty(r_values)
        error("No valid data points found in $filename")
    end

    if !issorted(r_values)
        println("Potential data r_values are not sorted. Sorting them now.")
        perm = sortperm(r_values)
        r_values = r_values[perm]
        potential_values = potential_values[perm]
    end

    println("Loaded potential data from $filename: $(length(r_values)) points")
    return PotentialData(r_values, potential_values)
end

function create_potential_lookup(potential_data::PotentialData)::PotentialLookup
    n_interp_points_between::Int = 1 # Set your desired value here

    orig_r = potential_data.r_values
    orig_v = potential_data.potential_values
    n_orig_points = length(orig_r)

    if n_orig_points < 2
        error("Potential data must contain at least 2 points for interpolation.")
    end
    if !issorted(orig_r)
        error("Input potential data r_values must be sorted for create_potential_lookup.")
    end

    n_lookup_points = n_orig_points + (n_orig_points - 1) * n_interp_points_between
    lookup_r = Vector{Float64}(undef, n_lookup_points)
    lookup_v = Vector{Float64}(undef, n_lookup_points)

    current_index = 1

    for i in 1:(n_orig_points - 1)
        r1, v1 = orig_r[i], orig_v[i]
        r2, v2 = orig_r[i + 1], orig_v[i + 1]

        lookup_r[current_index] = r1
        lookup_v[current_index] = v1
        current_index += 1

        if n_interp_points_between > 0
            if r1 == r2
                @warn "Duplicate r values found ($r1) at indices $i, $(i+1). Skipping interpolation between them."
                continue
            end
            delta_r_step = (r2 - r1) / (n_interp_points_between + 1)
            delta_v_step = (v2 - v1) / (n_interp_points_between + 1)

            for j in 1:n_interp_points_between
                lookup_r[current_index] = r1 + j * delta_r_step
                lookup_v[current_index] = v1 + j * delta_v_step
                current_index += 1
            end
        end
    end

    lookup_r[current_index] = orig_r[n_orig_points]
    lookup_v[current_index] = orig_v[n_orig_points]

    @assert current_index==n_lookup_points "Internal error: Index mismatch after interpolation."

    r_min = lookup_r[1]
    r_max = lookup_r[end]
    r_step = (n_lookup_points > 1) ? (r_max - r_min) / (n_lookup_points - 1) : 0.0

    println("Created potential lookup table with $n_lookup_points points (interpolation factor: $n_interp_points_between).")
    println(@sprintf("Lookup Parameters: r_min=%.6f, r_max=%.6f, calculated r_step=%.8f", r_min, r_max, r_step))

    return PotentialLookup(lookup_r, lookup_v, r_step, r_min, r_max)
end

function interpolate_potential(potential_data::PotentialData, r::Float64)::Float64
    r_values = potential_data.r_values
    potential_values = potential_data.potential_values
    n = length(r_values)

    if r <= r_values[1]
        return potential_values[1]
    elseif r >= r_values[n]
        return potential_values[n]
    end

    idx = searchsortedfirst(r_values, r)

    if r_values[idx] == r
        return potential_values[idx]
    else
        @assert idx>1 "Interpolation index error: idx=$idx for r=$r"
        r1, r2 = r_values[idx - 1], r_values[idx]
        v1, v2 = potential_values[idx - 1], potential_values[idx]
        if r1 == r2
            return v1
        end
        t = (r - r1) / (r2 - r1)
        return v1 + t * (v2 - v1)
    end
end

@inline function fast_interpolate_potential(r::Float64, lookup::PotentialLookup)::Float64
    if r <= lookup.r_min
        return lookup.lookup_v[1]
    elseif r >= lookup.r_max
        return lookup.lookup_v[end]
    end

    idx_f = (r - lookup.r_min) / lookup.r_step + 1.0 + 1e-9
    idx = floor(Int, idx_f)
    idx = max(1, min(idx, length(lookup.lookup_r) - 1))
    t = idx_f - idx

    v1 = lookup.lookup_v[idx]
    v2 = lookup.lookup_v[idx + 1]
    return v1 + t * (v2 - v1)
end

function cache_trajectory(sys_params::SystemParameters)::CachedTrajectory
    println("Caching trajectory for system: $(sys_params.system_name)")
    traj = read_xtc(sys_params)
    n_frames_total = Int(size(traj)) - 1
    start_frame_idx = 1
    n_frames_to_read = n_frames_total - (start_frame_idx - 1)

    frames = Vector{Frame}(undef, n_frames_to_read)
    boxes = Vector{Vector{Float64}}(undef, n_frames_to_read)

    println("Reading $(n_frames_to_read) frames...")
    try
        for i in 1:n_frames_to_read
            frame_idx_in_file = i + start_frame_idx - 1
            current_frame = read_step(traj, frame_idx_in_file)
            frames[i] = deepcopy(current_frame)
            boxes[i] = lengths(UnitCell(current_frame))
        end
    finally
        close(traj)
    end

    println("Cached $(n_frames_to_read) frames for system: $(sys_params.system_name)")
    return CachedTrajectory(frames, n_frames_to_read, boxes)
end

function read_random_frame_cached(cached_traj::CachedTrajectory, rng::Xoroshiro128Plus)
    frame_id = rand(rng, 1:(cached_traj.n_frames))
    frame = deepcopy(cached_traj.frames[frame_id])
    box = copy(cached_traj.boxes[frame_id])
    return frame, frame_id, box
end

function compute_potential_energy(distance_matrix::Matrix{Float64},
                                  lookup::PotentialLookup,
                                  sys_params::SystemParameters)::Float64
    total_energy = 0.0
    n_atoms = sys_params.n_atoms
    cutoff = lookup.r_max

    for i in 1:n_atoms
        for j in (i + 1):n_atoms
            r = distance_matrix[i, j]
            if lookup.r_min < r <= cutoff
                pair_energy = fast_interpolate_potential(r, lookup)
                total_energy += pair_energy
            end
        end
    end
    return total_energy
end

function create_magic_reference_data(input::MagicPreComputedInput,
                                     potential_data::PotentialData,
                                     lookup::PotentialLookup)::Tuple{MagicReferenceData, CachedTrajectory}
    nn_params = input.nn_params
    sys_params = input.system_params

    cached_traj = cache_trajectory(sys_params)
    n_frames = cached_traj.n_frames
    if n_frames == 0
        error("Cannot create reference data: Cached trajectory for $(sys_params.system_name) has 0 frames.")
    end

    distance_matrices = Vector{Matrix{Float64}}(undef, n_frames)
    g2_matrices = Vector{Matrix{Float64}}(undef, n_frames)
    potential_per_frame = Vector{Float64}(undef, n_frames)

    println("Precomputing reference data for $n_frames frames...")
    for frame_id in 1:n_frames
        frame = cached_traj.frames[frame_id]
        box = cached_traj.boxes[frame_id]

        dist_mat = build_distance_matrix(frame)
        distance_matrices[frame_id] = dist_mat
        potential_per_frame[frame_id] = compute_potential_energy(dist_mat, lookup, sys_params)
        g2_matrices[frame_id] = build_g2_matrix(dist_mat, nn_params)
    end
    println("Finished precomputing reference data.")

    return MagicReferenceData(distance_matrices, potential_per_frame, g2_matrices), cached_traj
end

function magic_single_particle_move!(ref_data::MagicReferenceData,
                                     model::Flux.Chain,
                                     nn_params::NeuralNetParameters,
                                     sys_params::SystemParameters,
                                     lookup::PotentialLookup,
                                     cached_traj::CachedTrajectory,
                                     rng::Xoroshiro128Plus)
    frame, frame_id, box = read_random_frame_cached(cached_traj, rng)
    point_index = rand(rng, 1:(sys_params.n_atoms))

    distance_matrix1 = ref_data.distance_matrices[frame_id]
    symm1 = ref_data.g2_matrices[frame_id]
    e_nn1_vector = init_system_energies_vector(symm1, model)
    e_nn1 = sum(e_nn1_vector)
    e_pot1 = ref_data.precomputed_energy[frame_id]
    distance_vec1 = distance_matrix1[:, point_index]

    original_pos = copy(positions(frame)[:, point_index])
    dr = sys_params.max_displacement * (rand(rng, Float64, 3) .- 0.5) * 2.0
    positions(frame)[:, point_index] .+= dr
    apply_periodic_boundaries!(frame, box, point_index)

    point_pos2 = positions(frame)[:, point_index]
    distance_vec2 = compute_distance_vector(point_pos2, positions(frame), box)

    distance_matrix2 = copy(distance_matrix1)
    distance_matrix2[point_index, :] = distance_vec2
    distance_matrix2[:, point_index] = distance_vec2

    e_pot2 = compute_potential_energy(distance_matrix2, lookup, sys_params)

    update_mask = get_energies_update_mask(distance_vec2, nn_params)
    g2_matrix2 = copy(symm1)
    update_g2_matrix!(g2_matrix2, distance_vec1, distance_vec2, sys_params, nn_params, point_index)
    symm2 = g2_matrix2

    e_nn2_vector = update_system_energies_vector(symm2, model, update_mask, e_nn1_vector)
    e_nn2 = sum(e_nn2_vector)

    return (symm1=symm1, symm2=symm2, Δe_nn=e_nn2 - e_nn1, Δe_pot=e_pot2 - e_pot1,
            e_nn1=e_nn1, e_pot1=e_pot1, e_nn2=e_nn2, e_pot2=e_pot2)
end

function magic_all_particle_move!(ref_data::MagicReferenceData,
                                  model::Flux.Chain,
                                  nn_params::NeuralNetParameters,
                                  sys_params::SystemParameters,
                                  lookup::PotentialLookup,
                                  cached_traj::CachedTrajectory,
                                  rng::Xoroshiro128Plus)
    frame, frame_id, box = read_random_frame_cached(cached_traj, rng)

    symm1 = ref_data.g2_matrices[frame_id]
    e_nn1_vector = init_system_energies_vector(symm1, model)
    e_nn1 = sum(e_nn1_vector)
    e_pot1 = ref_data.precomputed_energy[frame_id]

    dr = sys_params.max_displacement * (rand(rng, Float64, 3, sys_params.n_atoms) .- 0.5) * 2.0
    positions(frame) .+= dr
    for i in 1:(sys_params.n_atoms)
        apply_periodic_boundaries!(frame, box, i)
    end

    distance_matrix2 = build_distance_matrix(frame)
    e_pot2 = compute_potential_energy(distance_matrix2, lookup, sys_params)
    g2_matrix2 = build_g2_matrix(distance_matrix2, nn_params)
    symm2 = g2_matrix2
    e_nn2_vector = init_system_energies_vector(symm2, model)
    e_nn2 = sum(e_nn2_vector)

    return (symm1=symm1, symm2=symm2, Δe_nn=e_nn2 - e_nn1, Δe_pot=e_pot2 - e_pot1,
            e_nn1=e_nn1, e_pot1=e_pot1, e_nn2=e_nn2, e_pot2=e_pot2)
end

function make_magic_mc_move!(use_all_particles::Bool,
                             ref_data::MagicReferenceData,
                             model::Chain,
                             nn_params::NeuralNetParameters,
                             sys_params::SystemParameters,
                             lookup::PotentialLookup,
                             cached_traj::CachedTrajectory,
                             rng::Xoroshiro128Plus)
    if use_all_particles
        return magic_all_particle_move!(ref_data, model, nn_params, sys_params, lookup, cached_traj, rng)
    else
        return magic_single_particle_move!(ref_data, model, nn_params, sys_params, lookup, cached_traj, rng)
    end
end

function run_magic_training_phase!(;
                                   steps::Int,
                                   batch_size::Int,
                                   use_all_particles::Bool,
                                   system_params_list::Vector{SystemParameters},
                                   ref_data_list::Vector{MagicReferenceData},
                                   lookup_list::Vector{PotentialLookup},
                                   cached_traj_list::Vector{CachedTrajectory},
                                   model::Chain,
                                   nn_params::NeuralNetParameters,
                                   pretrain_params::PreTrainingParameters,
                                   optimizer,
                                   lr_schedule::Dict{Int, Float64},
                                   initial_lr::Float64,
                                   log_prefix::String="magic_training")
    rng = Xoroshiro128Plus()
    n_systems = length(system_params_list)
    opt_state = Flux.setup(optimizer, model)
    current_lr = initial_lr
    Flux.adjust!(opt_state, current_lr)

    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    phase_type = "abs"
    move_type = use_all_particles ? "all" : "single"
    summary_log_file = "$(log_prefix)_$(phase_type)_$(move_type)_$(timestamp)_summary.dat"

    println("Starting Magic Pre-Training Phase:")
    println("  Mode: Absolute Energy ($(phase_type)), Move Type: $(move_type)")
    println("  Epochs: $steps, Batch Size (Moves per system): $batch_size, Systems: $n_systems")
    println("  Initial Learning Rate: $initial_lr")
    println("  Summary Log: $summary_log_file")

    summary_io = open(summary_log_file, "w")
    println(summary_io, "# epoch mean_mae_diff mean_mse_diff mean_mae_abs mean_mse_abs mean_reg_loss")
    flush(summary_io)

    try
        for epoch in 1:steps
            if haskey(lr_schedule, epoch)
                current_lr = lr_schedule[epoch]
                Flux.adjust!(opt_state, current_lr)
            end

            accum_mse_diff = 0.0
            accum_mae_diff = 0.0
            accum_mse_abs = 0.0
            accum_mae_abs = 0.0
            accum_reg_loss = 0.0

            total_moves_in_epoch = 0

            epoch_flat_grad_sum = nothing
            grad_restructure = nothing

            for sys_id in 1:n_systems
                system_ref_data = ref_data_list[sys_id]
                system_lookup = lookup_list[sys_id]
                system_cached_traj = cached_traj_list[sys_id]
                system_params = system_params_list[sys_id]

                for batch_iter in 1:batch_size
                    move = make_magic_mc_move!(use_all_particles,
                                               system_ref_data,
                                               model,
                                               nn_params,
                                               system_params,
                                               system_lookup,
                                               system_cached_traj,
                                               rng)

                    symm1, symm2 = move.symm1, move.symm2
                    Δe_nn, Δe_pot = move.Δe_nn, move.Δe_pot
                    e_nn1, e_pot1 = move.e_nn1, move.e_pot1
                    e_nn2, e_pot2 = move.e_nn2, move.e_pot2

                    grad = compute_gradient!(e_nn2, e_pot2, Δe_nn, Δe_pot,
                                             symm1, symm2, model,
                                             pretrain_params, nn_params,
                                             false)

                    if grad !== nothing
                        flat_grad, restructure_func = Flux.destructure(grad)
                        if epoch_flat_grad_sum === nothing
                            epoch_flat_grad_sum = flat_grad
                            grad_restructure = restructure_func
                        else
                            epoch_flat_grad_sum .+= flat_grad
                        end
                        total_moves_in_epoch += 1
                    end

                    diff_mse = (Δe_nn - Δe_pot)^2
                    diff_mae = abs(Δe_nn - Δe_pot)
                    abs_mse = (e_nn2 - e_pot2)^2
                    abs_mae = abs(e_nn2 - e_pot2)
                    reg_loss_approx = pretrain_params.regularization * sum(p -> sum(abs2, p), Flux.trainables(model))

                    accum_mse_diff += diff_mse
                    accum_mae_diff += diff_mae
                    accum_mse_abs += abs_mse
                    accum_mae_abs += abs_mae
                    accum_reg_loss += reg_loss_approx
                end
            end

            epoch_flat_grad_avg = epoch_flat_grad_sum ./ total_moves_in_epoch
            grad_norm = sqrt(sum(abs2.(epoch_flat_grad_avg)))
            final_grad = grad_restructure(epoch_flat_grad_avg)
            update_model!(model, opt_state, final_grad)

            mean_mse_diff = accum_mse_diff / total_moves_in_epoch
            mean_mae_diff = accum_mae_diff / total_moves_in_epoch
            mean_mse_abs = accum_mse_abs / total_moves_in_epoch
            mean_mae_abs = accum_mae_abs / total_moves_in_epoch
            mean_reg_loss = accum_reg_loss / total_moves_in_epoch

            println(summary_io,
                    @sprintf("%d %.8f %.8f %.8f %.8f %.2e",
                             epoch,
                             mean_mae_diff,
                             mean_mse_diff,
                             mean_mae_abs,
                             mean_mse_abs,
                             mean_reg_loss))
            println(@sprintf("Magic | %s | Epoch: %4d | Moves: %5d | Diff MAE: %8.2f | Abs MAE: %8.2f | LR: %.2e | ∇ norm: %8.2f",
                             move_type, epoch, total_moves_in_epoch,
                             mean_mae_diff, mean_mae_abs, current_lr, grad_norm))

            flush(summary_io)

            if epoch % 50 == 0 || epoch == steps
                save_path = "magic-model-epoch-$(epoch).bson"
                @save save_path model
            end
        end
    finally
        close(summary_io)
        println("Log files closed.")
    end

    return model, opt_state
end

function run_magic_scan_training_phase!(;
                                        steps::Int,
                                        batch_size::Int,
                                        system_params_list::Vector{SystemParameters},
                                        ref_data_list::Vector{MagicReferenceData},
                                        lookup_list::Vector{PotentialLookup},
                                        cached_traj_list::Vector{CachedTrajectory},
                                        model::Chain,
                                        nn_params::NeuralNetParameters,
                                        pretrain_params::PreTrainingParameters,
                                        optimizer,
                                        lr_schedule::Dict{Int, Float64},
                                        initial_lr::Float64,
                                        scan_alpha::Float64,
                                        scan_steps::Int,
                                        log_prefix::String="magic_scan_pretraining")
    rng = Xoroshiro128Plus()
    n_systems = length(system_params_list)
    opt_state = Flux.setup(optimizer, model)
    current_lr = initial_lr
    Flux.adjust!(opt_state, current_lr)

    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    phase_type = "abs"
    summary_log_file = "$(log_prefix)_$(phase_type)_$(timestamp)_summary.dat"

    scan_points_per_axis = 2 * scan_steps
    total_scans_per_particle = 3 * scan_points_per_axis
    expected_max_moves_per_epoch = n_systems * batch_size * total_scans_per_particle

    println("Starting Magic Scan Pre-Training Phase:")
    println("  Mode: Absolute Energy ($(phase_type))")
    println("  Epochs: $steps, Batch Size: $batch_size, Systems: $n_systems")
    println("  Scan Range: +/- $(scan_alpha), Scan steps: $scan_steps => $total_scans_per_particle scans in all directions")
    println("  (Expected Moves: $expected_max_moves_per_epoch) = systems * batch_size * 3 axis * 2 directions * scan steps")
    println("  Initial Learning Rate: $initial_lr")
    println("  Summary Log: $summary_log_file")

    summary_io = open(summary_log_file, "w")
    println(summary_io, "# mean_mae_diff mean_mse_diff mean_mae_abs mean_mse_abs mean_reg_loss")
    flush(summary_io)

    try
        for epoch in 1:steps
            if haskey(lr_schedule, epoch)
                current_lr = lr_schedule[epoch]
                Flux.adjust!(opt_state, current_lr)
            end
            epoch_mse_diff = 0.0
            epoch_mae_diff = 0.0
            epoch_mse_abs = 0.0
            epoch_mae_abs = 0.0
            epoch_reg_loss = 0.0
            epoch_moves_processed = 0
            total_epoch_flat_grad = nothing
            grad_restructure = nothing

            step_size = scan_alpha / scan_steps
            scan_offsets = vcat(range(-scan_alpha, -step_size, length=scan_steps),
                                range(step_size, scan_alpha, length=scan_steps))
            if isempty(scan_offsets) && scan_steps > 0
                error("Scan offsets calculation failed.")
            end

            for sys_id in 1:n_systems
                system_ref_data = ref_data_list[sys_id]
                system_lookup = lookup_list[sys_id]
                system_cached_traj = cached_traj_list[sys_id]
                sys_params = system_params_list[sys_id]

                if sys_params.n_atoms == 0 || system_cached_traj.n_frames == 0
                    println("Warning: Skipping system $sys_id (atoms=$(sys_params.n_atoms), frames=$(system_cached_traj.n_frames)).")
                    continue
                end

                for particle_batch_iter in 1:batch_size
                    point_index = rand(rng, 1:(sys_params.n_atoms))
                    frame_work, frame_id, box = read_random_frame_cached(system_cached_traj, rng)

                    if frame_id < 1 || frame_id > length(system_ref_data.distance_matrices)
                        @warn "Invalid frame_id $frame_id for system $sys_id. Skipping particle scan."
                        continue
                    end
                    dist_matrix_orig = system_ref_data.distance_matrices[frame_id]
                    g2_matrix_orig = system_ref_data.g2_matrices[frame_id]
                    dist_vec_orig = dist_matrix_orig[:, point_index]

                    orig_coord = copy(positions(frame_work)[:, point_index])
                    initial_dr = (sys_params.max_displacement) * (rand(rng, Float64, 3) .- 0.5) * 2.0
                    positions(frame_work)[:, point_index] .+= initial_dr
                    apply_periodic_boundaries!(frame_work, box, point_index)
                    perturbed_coord = copy(positions(frame_work)[:, point_index])

                    dist_vec_1_prime = compute_distance_vector(perturbed_coord, positions(frame_work), box)
                    dist_matrix_1_prime = copy(dist_matrix_orig)
                    dist_matrix_1_prime[point_index, :] = dist_vec_1_prime
                    dist_matrix_1_prime[:, point_index] = dist_vec_1_prime

                    e_pot_1_prime = compute_potential_energy(dist_matrix_1_prime, system_lookup, sys_params)
                    g2_matrix_1_prime = copy(g2_matrix_orig)
                    update_g2_matrix!(g2_matrix_1_prime, dist_vec_orig, dist_vec_1_prime, sys_params, nn_params,
                                      point_index)
                    e_nn_1_prime = sum(init_system_energies_vector(g2_matrix_1_prime, model))

                    for axis in 1:3
                        for offset in scan_offsets
                            dr_scan = zeros(Float64, 3)
                            dr_scan[axis] = offset
                            positions(frame_work)[:, point_index] = perturbed_coord .+ dr_scan
                            apply_periodic_boundaries!(frame_work, box, point_index)
                            scan_coord = positions(frame_work)[:, point_index]

                            dist_vec_2 = compute_distance_vector(scan_coord, positions(frame_work), box)
                            dist_matrix_2 = copy(dist_matrix_1_prime)
                            dist_matrix_2[point_index, :] = dist_vec_2
                            dist_matrix_2[:, point_index] = dist_vec_2

                            e_pot_2 = compute_potential_energy(dist_matrix_2, system_lookup, sys_params)
                            g2_matrix_2 = copy(g2_matrix_1_prime)
                            update_g2_matrix!(g2_matrix_2, dist_vec_1_prime, dist_vec_2, sys_params, nn_params,
                                              point_index)
                            e_nn_2 = sum(init_system_energies_vector(g2_matrix_2, model))

                            Δe_nn_scan = e_nn_2 - e_nn_1_prime
                            Δe_pot_scan = e_pot_2 - e_pot_1_prime

                            grad = compute_gradient!(e_nn_2, e_pot_2, Δe_nn_scan, Δe_pot_scan,
                                                     g2_matrix_1_prime, g2_matrix_2, model,
                                                     pretrain_params, nn_params,
                                                     false)

                            if !isnothing(grad)
                                flat_grad, restructure_func = Flux.destructure(grad)
                                if isnothing(total_epoch_flat_grad)
                                    total_epoch_flat_grad = flat_grad
                                    grad_restructure = restructure_func
                                else
                                    total_epoch_flat_grad .+= flat_grad
                                end
                                epoch_moves_processed += 1
                            end

                            diff_mae = abs(Δe_nn_scan - Δe_pot_scan)
                            diff_mse = (Δe_nn_scan - Δe_pot_scan)^2
                            abs_mae = abs(e_nn_2 - e_pot_2)
                            abs_mse = (e_nn_2 - e_pot_2)^2
                            reg_loss_approx = pretrain_params.regularization *
                                              sum(p -> sum(abs2, p), Flux.trainables(model))
                            epoch_mae_diff += diff_mae
                            epoch_mse_diff += diff_mse
                            epoch_mae_abs += abs_mae
                            epoch_mse_abs += abs_mse
                            epoch_reg_loss += reg_loss_approx
                        end
                    end
                end
            end

            total_epoch_flat_grad ./= epoch_moves_processed
            grad_norm = sqrt(sum(abs2.(total_epoch_flat_grad)))
            final_grad = grad_restructure(total_epoch_flat_grad)
            update_model!(model, opt_state, final_grad)

            mean_mae_diff = epoch_mae_diff / epoch_moves_processed
            mean_mse_diff = epoch_mse_diff / epoch_moves_processed
            mean_mae_abs = epoch_mae_abs / epoch_moves_processed
            mean_mse_abs = epoch_mse_abs / epoch_moves_processed
            mean_reg_loss = epoch_reg_loss / epoch_moves_processed

            println(summary_io,
                    @sprintf("%d %.8f %.8f %.8f %.8f %.2e", epoch,
                             mean_mae_diff, mean_mse_diff, mean_mae_abs, mean_mse_abs, mean_reg_loss))
            println(@sprintf("Magic Scan | %s | Epoch: %4d | Batch Size: %d | Moves: %7d | Diff MAE: %8.2f | Abs MAE: %8.2f | LR: %.2e | ∇ norm: %8.2f",
                             phase_type, epoch, batch_size, epoch_moves_processed, mean_mae_diff, mean_mae_abs,
                             current_lr, grad_norm))

            flush(summary_io)

            if epoch % 50 == 0 || epoch == steps
                save_path = "magic-scan-model-epoch-$(epoch).bson"
                @save save_path model
            end
        end
    finally
        close(summary_io)
    end
    return model, opt_state
end

function magic_pretrain(potential_files::Vector{String},
                        pretrain_params::PreTrainingParameters,
                        nn_params::NeuralNetParameters,
                        system_params_list::Vector{SystemParameters},
                        model::Chain,
                        optimizer,
                        reference_rdfs::Vector{Vector{Float64}})
    println("--- Starting Magic Pre-Training Setup ---")
    n_systems = length(system_params_list)
    if length(potential_files) != n_systems || length(reference_rdfs) != n_systems
        error("Input list lengths mismatch.")
    end

    println("Loading potential data...")
    potential_data_list = [load_potential_data(file) for file in potential_files]

    println("Creating potential lookup tables...")
    lookup_list = [create_potential_lookup(data) for data in potential_data_list]

    println("Creating reference data and caching trajectories...")
    ref_inputs = [MagicPreComputedInput(nn_params, system_params_list[i], reference_rdfs[i])
                  for i in 1:n_systems]

    ref_data_list = Vector{MagicReferenceData}(undef, n_systems)
    cached_traj_list = Vector{CachedTrajectory}(undef, n_systems)

    for i in 1:n_systems
        println("\nProcessing system $(i) / $n_systems: $(system_params_list[i].system_name)")
        ref_data, cached_traj = create_magic_reference_data(ref_inputs[i], potential_data_list[i], lookup_list[i])
        ref_data_list[i] = ref_data
        cached_traj_list[i] = cached_traj
        println("System $i reference data created and trajectory cached.")
    end

    println("\n--- Configuring Magic MC Training Parameters ---")
    training_steps = 10000
    training_batch_size = 50
    use_all_particles_move = false

    lr_schedule = Dict(1 => 0.000001,
                       10 => 0.00002,
                       50 => 0.00001,
                       100 => 0.000005,
                       500 => 0.000002,
                       800 => 0.000001)
    initial_training_lr = pretrain_params.learning_rate

    FILE_TO_LOAD = ""
    if !isempty(FILE_TO_LOAD) && isfile(FILE_TO_LOAD)
        try
            @load FILE_TO_LOAD model
            println("Successfully loaded model from $(FILE_TO_LOAD)")
        catch e
            @warn "Could not load model $(FILE_TO_LOAD). Starting from initial model. Error: $e"
        end
    else
        println("No model file specified or found ('$(FILE_TO_LOAD)'). Starting training from initial model.")
    end

    println("\n--- Starting Magic MC Training Phase ---")
    model, opt_state = run_magic_training_phase!(steps=training_steps,
                                                 batch_size=training_batch_size,
                                                 use_all_particles=use_all_particles_move,
                                                 system_params_list=system_params_list,
                                                 ref_data_list=ref_data_list,
                                                 lookup_list=lookup_list,
                                                 cached_traj_list=cached_traj_list,
                                                 model=model,
                                                 nn_params=nn_params,
                                                 pretrain_params=pretrain_params,
                                                 optimizer=optimizer,
                                                 lr_schedule=lr_schedule,
                                                 initial_lr=initial_training_lr,
                                                 log_prefix="magic_pt")

    println("\n--- Saving Final Model from MC Phase ---")
    mode_str = "abs"
    move_str = use_all_particles_move ? "all" : "single"
    final_save_path = "magic-pt-final-model-$(mode_str)-$(move_str).bson"
    @save final_save_path model
    println("Final MC phase model saved to $(final_save_path)")

    println("\n--- Magic pre-training (magic_pretrain) completed! ---")
    return model
end

function magic_scan_pretrain(potential_files::Vector{String},
                             pretrain_params::PreTrainingParameters,
                             nn_params::NeuralNetParameters,
                             system_params_list::Vector{SystemParameters},
                             model::Chain,
                             optimizer,
                             reference_rdfs::Vector{Vector{Float64}})
    println("--- Starting Magic Scan Pre-Training Setup ---")
    n_systems = length(system_params_list)
    if length(potential_files) != n_systems || length(reference_rdfs) != n_systems
        error("Input list lengths mismatch.")
    end

    println("Loading potential data...")
    potential_data_list = [load_potential_data(file) for file in potential_files]

    println("Creating potential lookup tables...")
    lookup_list = [create_potential_lookup(data) for data in potential_data_list]

    println("Creating reference data and caching trajectories...")
    ref_inputs = [MagicPreComputedInput(nn_params, system_params_list[i], reference_rdfs[i])
                  for i in 1:n_systems]
    ref_data_list = Vector{MagicReferenceData}(undef, n_systems)
    cached_traj_list = Vector{CachedTrajectory}(undef, n_systems)
    for i in 1:n_systems
        println("\nProcessing system $(i) / $n_systems: $(system_params_list[i].system_name)")
        ref_data, cached_traj = create_magic_reference_data(ref_inputs[i], potential_data_list[i], lookup_list[i])
        ref_data_list[i] = ref_data
        cached_traj_list[i] = cached_traj
        println("System $i reference data created and trajectory cached.")
    end

    println("\n--- Configuring Magic Scan Training Parameters ---")
    training_steps = 1000
    training_batch_size = 10
    scan_distance = 0.5
    scan_num_steps = 10

    lr_schedule = Dict(1 => 0.000001,
                       10 => 0.00002,
                       50 => 0.00001,
                       100 => 0.000005,
                       500 => 0.000002,
                       800 => 0.000001)
    initial_training_lr = pretrain_params.learning_rate

    FILE_TO_LOAD = ""
    if !isempty(FILE_TO_LOAD) && isfile(FILE_TO_LOAD)
        try
            @load FILE_TO_LOAD model
            println("Successfully loaded model from $(FILE_TO_LOAD) for scan phase.")
        catch e
            @warn "Could not load model $(FILE_TO_LOAD). Starting scan phase from initial model. Error: $e"
        end
    else
        println("No model file specified or found ('$(FILE_TO_LOAD)'). Starting scan phase from initial model.")
    end

    println("\n--- Starting Magic Scan Training Phase ---")
    model, opt_state = run_magic_scan_training_phase!(steps=training_steps,
                                                      batch_size=training_batch_size,
                                                      scan_alpha=scan_distance,
                                                      scan_steps=scan_num_steps,
                                                      system_params_list=system_params_list,
                                                      ref_data_list=ref_data_list,
                                                      lookup_list=lookup_list,
                                                      cached_traj_list=cached_traj_list,
                                                      model=model,
                                                      nn_params=nn_params,
                                                      pretrain_params=pretrain_params,
                                                      optimizer=optimizer,
                                                      lr_schedule=lr_schedule,
                                                      initial_lr=initial_training_lr,
                                                      log_prefix="magic_scan_pt")

    println("\n--- Saving Final Scan Model ---")
    mode_str = "abs"
    final_save_path = "magic-scan-pt-final-model-$(mode_str).bson"
    @save final_save_path model

    println("Final scan phase model saved to $(final_save_path)")

    println("\n--- Magic scan pre-training (magic_scan_pretrain) completed! ---")
    return model
end
