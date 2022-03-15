using Revise
using POMDPs
using POMDPModelTools
using POMDPPolicies
using POMDPSimulators
using POMDPPolicies
using BeliefUpdaters
using ParticleFilters
using Reel
using Distributions

include("./BasicPOMCP/src/BasicPOMCP.jl")
using .BasicPOMCP

using D3Trees

# Load the maze matrix into a julia array
function read_grid(filename::String)
    lines = readlines(filename)
    grid_lines = lines[3:length(lines), :]
    grid = zeros(length(grid_lines), length(grid_lines[1]))
   
    for i in eachindex(grid_lines)
        cells = split(grid_lines[i], "")
        for (j, cell) in enumerate(cells)
            grid[i, j] = parse(Int64, cell)
        end
    end
    return grid 
end

function get_symbol(cell::Int64)
    if(cell == 0)
        return (:empty)
    elseif(cell == 2)
        return (:reward)
    elseif(cell == 5)
        return (:agent_up)
    elseif(cell == 1)
        return (:agent_right)
    elseif(cell == 4)
        return (:agent_down)
    elseif(cell == 9)
        return (:agent_left)
    elseif(cell == 6)
        return (:unobserved)
    elseif(cell == 3)
        return (:wall_c0)
    elseif(cell == 7)
        return (:wall_c1)
    elseif(cell == 8)
        return (:wall_c2)
    end
end


function is_reward(d::Symbol)
    return d == :reward
end

function is_reward(rd::Int64)
    d = get_symbol(rd) 
    return is_reward(d)
end


function is_agent(d::Symbol)
    return d == :agent_up || d == :agent_down || d == :agent_right || d == :agent_left
end

function is_agent(rd::Int64)
    d = get_symbol(rd) 
    return is_agent(d)
end


function get_direction_meaning(sd::Symbol)
    if(sd == :agent_up)
        return (-1, 0)
    elseif(sd == :agent_right)
        return (0, 1)
    elseif(sd == :agent_down)
        return (1, 0)
    elseif(sd == :agent_left)
        return (0, -1)
    end
end

function get_direction_meaning(d::Int64)
    return get_direction_meaning(get_symbol(d))
end

function get_symbol(cell::Symbol)
    if(cell ==  :empty)
        return 0  
    elseif(cell == :reward)
        return 2
    elseif(cell == :agent_up)
        return 5
    elseif(cell == :agent_right)
        return 1
    elseif(cell == :agent_down)
        return 4
    elseif(cell == :agent_left)
        return 9
    elseif(cell == :unobserved)
        return 6
    elseif(cell == :wall_c0) # Three different wall colors
        return 3
    elseif(cell == :wall_c1)
        return 7
    elseif(cell == :wall_c2)
        return 8
    end
end


function is_wall(cell::Symbol)
    return (cell == :wall_c0 || cell == :wall_c1 || cell == :wall_c2)
end

function is_wall(cell::Int64)
    return is_wall(get_symbol(cell))
end



struct GridWorldObs
    ay::Int64
    ax::Int64
end

struct GridWorldState
    grid::Matrix{Int64}
    ay::Int64
    ax::Int64
    d::Int64
    done::Bool # are we in a terminal state?
    probs::Vector{Float64}
    reward_count::Int64
    reward_status::Vector{Vector{GridWorldObs}}
    obs_reward_map::Matrix{Float64}
    obs_reward_key::Vector{GridWorldObs}
end

function GridWorldState(grid::Matrix{Int64},
                        ay::Int64,
                        ax::Int64,
                        d::Int64,
                        done::Bool,
                        probs::Vector{Float64},
                        reward_count::Int64,
                        reward_status::Vector{Vector{GridWorldObs}})
    # These are dummy
    sy, sx = size(grid)
    obs_reward_map = ones(Int64, sy, sx)*(get_symbol(:unobserved))
    obs_reward_key = GridWorldObs[]

    return GridWorldState(grid, ay, ax, d, done, probs, reward_count, reward_status, obs_reward_map, obs_reward_key)
end



mutable struct GridWorldSSP <: POMDP{GridWorldState, Symbol, GridWorldState}
    size_y::Int64 # x size of the grid
    size_x::Int64 # y size of the grid
    observation_mode::Symbol
    real_observed_state::GridWorldState
    hyps::Vector{Matrix{Int64}}
    reward_maps::Vector{Vector{Matrix{Float64}}}
    discount_factor::Float64
    optimism::Float64
    room_keys::Matrix{Int64}
    room_key_masks::Vector{BitMatrix}
end

# checks if the position of two states are the same
# posequal(s1::GridWorldState, s2::GridWorldState) = s1.x == s2.x && s1.y == s2.y

# the grid world mdp type
mutable struct GridWorld <: POMDP{GridWorldState, Symbol, GridWorldState}
    size_y::Int64 # x size of the grid
    size_x::Int64 # y size of the grid
    observation_mode::Symbol
    real_observed_state::GridWorldState
    hyps::Vector{Matrix{Int64}}
    discount_factor::Float64
    optimism::Float64
    room_keys::Matrix{Int64}
    room_key_masks::Vector{BitMatrix}
end

function get_rotation(d::Int64, a::String)
    return get_rotation(get_symbol(d), Symbol(a))
end



function get_rotation(d::Symbol, a::Symbol)
    if(a == :rotate_left)
        if(d == :agent_up)
            return get_symbol(:agent_left)
        elseif(d == :agent_left)
            return get_symbol(:agent_down)
        elseif(d == :agent_down)
            return get_symbol(:agent_right)
        elseif(d == :agent_right)
            return get_symbol(:agent_up)
        end
    elseif(a == :rotate_right)
        if(d == :agent_up)
            return get_symbol(:agent_right)
        elseif(d == :agent_left)
            return get_symbol(:agent_up)
        elseif(d == :agent_down)
            return get_symbol(:agent_left)
        elseif(d == :agent_right)
            return get_symbol(:agent_down)
        end
    end
end

function absolute_to_relative(absolute_action::Int64, absolute_rot::Int64)
    return absolute_to_relative(get_symbol(absolute_action), get_symbol(absolute_rot))
end

function absolute_to_relative(absolute_action::Symbol, absolute_rot::Symbol)
    """
     Takes in the absolute action and agent direction and outputs the relative agent action
    """
    if(absolute_action == :agent_up)
        if(absolute_rot == :agent_up)
            return :forward
        elseif(absolute_rot == :agent_left)
            return :right
        elseif(absolute_rot == :agent_down)
            return :backward
        elseif(absolute_rot == :agent_right)
            return :left
        end
    elseif(absolute_action == :agent_right)
        if(absolute_rot == :agent_up)
            return :right
        elseif(absolute_rot == :agent_left)
            return :backward
        elseif(absolute_rot == :agent_down)
            return :left
        elseif(absolute_rot == :agent_right)
            return :forward
        end
    elseif(absolute_action == :agent_left)
        if(absolute_rot == :agent_up)
            return :left
        elseif(absolute_rot == :agent_left)
            return :forward
        elseif(absolute_rot == :agent_down)
            return :right
        elseif(absolute_rot == :agent_right)
            return :backward
        end
    elseif(absolute_action == :agent_down)
        if(absolute_rot == :agent_up)
            return :backward
        elseif(absolute_rot == :agent_left)
            return :left
        elseif(absolute_rot == :agent_down)
            return :forward
        elseif(absolute_rot == :agent_right)
            return :right
        end
    end
end

function get_range(startc::Int64, endc::Int64)
    if(endc > startc)
        return collect(startc:endc)
    else
        return reverse(collect(endc:startc))
    end
end

function get_action_offset(y::Int64, x::Int64, d::Int64, a::Symbol)
    dy, dx = get_direction_meaning(d)
    if a == :forward
        return (get_range(y, y+dy), get_range(x, x+dx), d)
    elseif a == :forward5
        return (get_range(y, y+dy*5), get_range(x, x+dx*5), d)
    elseif a == :backward # not used by pomcp model
        return (get_range(y, y-dy), get_range(x, x-dx), d)
    elseif a == :backward5
        return (get_range(y, y-dy*5), get_range(x, x-dx*5), d)
    elseif a == :left
        return (get_range(y, y-dx), get_range(x, x+dy), d)
    elseif a == :left5
        return (get_range(y, y-dx*5), get_range(x, x+dy*5), d)
    elseif a == :right
        return (get_range(y, y+dx), get_range(x, x-dy), d)
    elseif a == :right5
        return (get_range(y, y+dx*5), get_range(x, x-dy*5), d)
    elseif a == :rotate_left || a == :rotate_right
        return (get_range(y, y), get_range(x, x), get_rotation(get_symbol(d), a)) # Rotate left by subtracting 1 from d
    end
end

function get_agent_loc(s::Matrix{Int64})
    for y in 1:size(s, 1) 
        for x in 1:size(s, 2)
            cell_sym = s[y, x]
            if(is_agent(cell_sym))
                return y, x, cell_sym
            end
        end
    end
    @assert false "Agent must be in grid"
end

function get_reward_loc(hyp::Matrix{Int64}, os::Matrix{Int64})
    obs_vec = []
    for y in 1:size(hyp, 1) 
        for x in 1:size(hyp, 2)
            if(is_reward(hyp[y, x]) && (os[y, x] == get_symbol(:unobserved) || os[y, x] == get_symbol(:reward)) )
                push!(obs_vec, GridWorldObs(y, x))
            end
        end
    end
    return obs_vec
end

function get_agent_loc(s::GridWorldState)
    return s.ay, s.ax, s.d
end

function get_theta_midpoint(sd::Int64)
    d = get_symbol(sd)
    if(d == :agent_up)
        return 270-45
    elseif(d == :agent_left)
        return 180-45
    elseif(d == :agent_down)
        return 90-45
    elseif(d == :agent_right)
        return 360-45
    end
end

function get_state_observation_mask(observed_state::GridWorldState, state_grid::Matrix{Int64}, observation_mode::Symbol, ay::Int64, ax::Int64, d::Int64, room_keys::Matrix{Int64}, room_key_masks::Vector{BitMatrix})
    sy, sx = size(state_grid)
    mask = falses(sy, sx)
    if(observation_mode == :fixed_radius)
        for y in max(1, ay-1):min(ay+1, sy)
            for x in max(ax-1, 1):min(ax+1, sx)
                if(observed_state.grid[y, x] == get_symbol(:unobserved))
                    mask[y, x] = 1
                end
            end
        end
    elseif(observation_mode == :fixed_radius_med)
        for y in max(1, ay-2):min(ay+2, sy)
            for x in max(ax-2, 1):min(ax+2, sx)
                if(observed_state.grid[y, x] == get_symbol(:unobserved))
                    mask[y, x] = 1
                end
            end
        end
    elseif(observation_mode == :room)
        mask = room_key_masks[room_keys[ay, ax]+1]

    elseif(observation_mode == :line_of_sight)
        ax = ax-0.5
        ay = ay-0.5
        theta_resolution = 5
        dist_resolution = 0.3
        render_dist = 5
        for theta in 0:theta_resolution:360
            for dist in 0:dist_resolution:min(max(sy, sx), render_dist)
                yi = convert(Int64, ceil(ay+dist*sin(theta)))
                xi = convert(Int64, ceil(ax+dist*cos(theta)))
                if(yi > 0 && yi <= sy && xi > 0 && xi <= sx)
                    if(observed_state.grid[yi, xi] == get_symbol(:unobserved))
                        mask[yi, xi] = 1
                    end
                    if ( is_wall(state_grid[yi, xi]) || state_grid[yi, xi] ==  get_symbol(:reward))
                        break
                    end
                else
                    break
                end
            end
        end
    elseif(observation_mode == :directional_line_of_sight)
        ax = ax-0.5
        ay = ay-0.5
        theta_resolution = 1
        dist_resolution = 0.5
        render_dist = 30
        midpoint = get_theta_midpoint(d)
        for theta_d in 0:theta_resolution:(360/4.0)
            theta_degrees = (theta_d+midpoint)%360
            theta = theta_degrees*pi/180
            for dist in 0:dist_resolution:min(max(sy, sx), render_dist)
                yi = convert(Int64, ceil(ay+dist*sin(theta)))
                xi = convert(Int64, ceil(ax+dist*cos(theta)))
                if(yi > 0 && yi <= sy && xi > 0 && xi <= sx)
                    if(observed_state.grid[yi, xi] == get_symbol(:unobserved))
                        mask[yi, xi] = 1
                    end
                    if ( is_wall(state_grid[yi, xi]) || state_grid[yi, xi] ==  get_symbol(:reward))
                        break
                    end
                else
                    break
                end
            end
        end
    end
    return mask
end


function has_wall(offys::Vector{Int64}, offxs::Vector{Int64}, state_grid::Matrix{Int64})
    for yi in 1:length(offys)
        for xi in 1:length(offxs)
            if(is_wall(state_grid[offys[yi], offxs[xi]]))
                return true
            end
        end
    end
    return false
end


function get_first_nonwall(offys::Vector{Int64}, offxs::Vector{Int64}, state_grid::Matrix{Int64}, size_y::Int64, size_x::Int64)
    offy, offx = offys[1], offxs[1]
    if(length(offys)>1)
        for yi in 1:length(offys)
            if ((offys[yi] >= 1) && (offys[yi] <= size_y) && (offxs[1] >= 1) && (offxs[1] <= size_x))
                if(is_wall(state_grid[offys[yi], offxs[1]]))
                    return offy, offx
                else
                    offy, offx = offys[yi], offxs[1]
                end
            else
                return offy, offx
            end
        end
    elseif(length(offxs)>1)
        for xi in 1:length(offxs)
            if ((offys[1] >= 1) && (offys[1] <= size_y) && (offxs[xi] >= 1) && (offxs[xi] <= size_x))
                if(is_wall(state_grid[offys[1], offxs[xi]]))
                    return offy, offx
                else
                    offy, offx = offys[1], offxs[xi]
                end
            else
                 return offy, offx
            end
        end
    end
    return offy, offx
end



function in_state_bounds(state, y, x)
    sy, sx = size(state)
    return y>=1 && x>=1 && y<=sy && x<=sx
end


function get_cardianals(y, x)
    return [
        (y-1, x+0),
        (y+1, x+0),
        (y+0, x+1),
        (y+0, x-1)
    ]
end

function find_reachable_unobserved(observation, ay, ax)
    Q = [GridWorldObs(ay, ax)]
    explored = Dict()
    while(length(Q) > 0)
        q = pop!(Q)
        y, x = q.ay, q.ax

        if(observation[y, x] == get_symbol(:unobserved))
            return y, x
        end

        for (ny, nx) in get_cardianals(y, x)
            neighbor = GridWorldObs(ny, nx)
            if(!get(explored, neighbor, false) && in_state_bounds(observation, ny, nx) && (observation[ny, nx] == get_symbol(:unobserved) || observation[ny, nx] == get_symbol(:empty)) )
                explored[neighbor] = true
                push!(Q, neighbor)
            end
        end
    end
    return -1, -1
end


function fill_obsmap(observation)
    sy, sx = size(observation)
    new_state = zeros(sy, sx)
    Q = []
    explored = Dict()
    for y in 1:sy
        for x in 1:sx
            if(observation[y, x] == get_symbol(:unobserved))
                push!(Q, (y, x, 0))
                explored[GridWorldObs(y, x)] = true
                new_state[y, x] = 1.0
            end
        end
    end

    frontier = [GridWorldObs(y, x) for (y, x, d) in Q]

    while(length(Q) > 0)
        q = pop!(Q)
        y, x, depth = q
        rval = 1.0/((depth)+1)
        if(rval > new_state[y, x])
            new_state[y, x] = rval
        end

        for (ny, nx) in get_cardianals(y, x)
            neighbor = (ny, nx, depth + 1)
            is_explored = get(explored, GridWorldObs(ny, nx), false)
            if(!is_explored && in_state_bounds(new_state, ny, nx) && (observation[ny, nx] == get_symbol(:empty) || observation[ny, nx] == get_symbol(:unobserved)))
                explored[GridWorldObs(ny, nx)] = true
                push!(Q, neighbor)
            end
        end
    end
    return new_state, frontier
end


function grid_tf(hyps::Vector{Matrix{Int64}}, reward_maps::Vector{Vector{Matrix{Float64}}}, s::GridWorldState, os::GridWorldState, a::Symbol, size_y::Int64, size_x::Int64, optimism::Float64)

    agent_loc_y, agent_loc_x, d = get_agent_loc(s)
    r = 0.0

    s_grid = copy(s.grid)
    os_grid = copy(os.grid)

    offys, offxs, offd = get_action_offset(agent_loc_y, agent_loc_x, d, a)
    offy, offx = get_first_nonwall(offys, offxs, s.grid, size_y, size_x)
    s_grid[agent_loc_y, agent_loc_x] = get_symbol(:empty)
    os_grid[agent_loc_y, agent_loc_x] = get_symbol(:empty)

    s_grid[offy, offx] = offd
    os_grid[offy, offx] = offd

    # Cross out the hypotheses that transition is in conflict with
    new_probs = copy(s.probs)./sum(s.probs) 
    aux_r = 0
    prob_power = 4

    @assert length(hyps) == length(reward_maps)

    if(!isnan(sum(new_probs)) && sum(new_probs) > 0.0)
        for hypi in 1:length(hyps)
            max_reward_map = 0
            for hypr in 1:length(s.reward_status[hypi])
                if( (s.reward_status[hypi][hypr].ay != -1) && (s.reward_status[hypi][hypr].ax != -1) 
                    && reward_maps[hypi][hypr][offy, offx] > max_reward_map)
                    max_reward_map = reward_maps[hypi][hypr][offy, offx]
                end
            end
            aux_r += max_reward_map * (new_probs[hypi]^prob_power)
        end
    end
    aux_r += s.obs_reward_map[offy, offx]
    
    if(get_symbol(s.grid[offy, offx]) == :reward)
        r = 1.0
    end

    total_rewards = s.reward_count
    if(r > 0)
        total_rewards = total_rewards+1
    end

    # Need to generate a new observation reward map
    new_obs_reward_map = s.obs_reward_map
    new_obs_reward_key = s.obs_reward_key

    term = (offy==agent_loc_y && offx==agent_loc_x && d==offd)
    return GridWorldState(s_grid, offy, offx, offd, term, new_probs, total_rewards, s.reward_status, new_obs_reward_map, new_obs_reward_key), 
           GridWorldState(os_grid, offy, offx, offd, term, new_probs, total_rewards, s.reward_status, new_obs_reward_map, new_obs_reward_key), 
           r+aux_r
end

function count_unobserved(grid::Matrix{Int64})
    return length(grid[grid .== get_symbol(:unobserved)])
end

function grid_tf(s::GridWorldState, os::GridWorldState, a::Symbol, size_y::Int64, size_x::Int64, optimism::Float64)
    return grid_tf(Matrix{Int64}[], Vector{Matrix{Float64}}[], s, os, a, size_y, size_x, optimism)
end

function get_unobserved(sy::Int64, sx::Int64, ay::Int64, ax::Int64, d::Int64)
    default_grid = ones(Int64, sy, sx)*(get_symbol(:unobserved))
    return GridWorldState(default_grid, ay, ax, d, false, Float64[],  0, Vector{GridWorldObs}[])
end

function update_observation_from_state(os::GridWorldState, s_grid::Matrix{Int64}, observation_mode::Symbol, room_keys::Matrix{Int64}, room_key_masks::Vector{BitMatrix})
    oscopy = deepcopy(os)
    ay, ax, d = get_agent_loc(oscopy)
    mask = get_state_observation_mask(oscopy, s_grid, observation_mode, ay, ax, d, room_keys, room_key_masks)
    oscopy.grid[mask] = s_grid[mask]
    return oscopy
end 

############################ POMDP Interface ####################################

function POMDPs.gen(m::GridWorld, s::GridWorldState, a::Symbol, rng) 
    # Select a possible next state
    sy, sx = size(s.grid)
    # num_rewards = sy*sx*m.optimism
    hyp_grid = deepcopy(s.grid)

    os, oe, or = get_symbol(:unobserved), get_symbol(:empty), get_symbol(:reward)
  
    hyp_grid[hyp_grid .== os] .=  get_symbol(:empty)

    hyp_state = GridWorldState(hyp_grid, s.ay, s.ax, copy(s.d), copy(s.done), deepcopy(s.probs), s.reward_count, s.reward_status, s.obs_reward_map, s.obs_reward_key)
    new_hypstate, new_observed_state, r = grid_tf(hyp_state, s, a, m.size_y, m.size_x, m.optimism)

    # Transision in the observed state wrt action
    sp = update_observation_from_state(new_observed_state, new_hypstate.grid, m.observation_mode, m.room_keys, m.room_key_masks)

    unobserved_count = count_unobserved(sp.grid)
    prev_unobserved_count = count_unobserved(s.grid)
    sp = GridWorldState(sp.grid, sp.ay, sp.ax, sp.d, sp.done, sp.probs, sp.reward_count, sp.reward_status, s.obs_reward_map, s.obs_reward_key)

    # observed_reward = (prev_unobserved_count-unobserved_count)*m.optimism
    return (sp=sp, o=deepcopy(sp), r=r)
end


function POMDPs.gen(m::GridWorldSSP, s::GridWorldState, a::Symbol, rng) 
    # if(length(s.probs) == 0 || sum(s.probs) == 0)
    sy, sx = size(s.grid)
    # num_rewards = sy*sx*m.optimism
    hyp_grid = copy(s.grid)

    os, oe, or = get_symbol(:unobserved), get_symbol(:empty), get_symbol(:reward)

    temp_probs = deepcopy(s.probs)

    while(length(temp_probs) != 0 && sum(temp_probs) != 0 && !isnan(sum(temp_probs)) && count_unobserved(hyp_grid) != 0)
    
        hyp_dist = Categorical(temp_probs./sum(temp_probs))
        # Replace all of the unobservred cells with their hypothesized values
        selection = rand(rng, hyp_dist)
        temp_probs[selection] = 0
        hyp_grid[hyp_grid .== os] = m.hyps[selection][hyp_grid .== os]
    end
   
    hyp_grid[hyp_grid .== os] .=  get_symbol(:empty)

    hyp_state = GridWorldState(hyp_grid, s.ay, s.ax, s.d, s.done, deepcopy(s.probs), s.reward_count, s.reward_status, s.obs_reward_map, s.obs_reward_key)
    new_hypstate, new_observed_state, r = grid_tf(m.hyps, m.reward_maps, hyp_state, s, a, m.size_y, m.size_x, m.optimism)

    # Transision in the observed state wrt action
    sp = update_observation_from_state(new_observed_state, new_hypstate.grid, m.observation_mode, m.room_keys, m.room_key_masks)

    new_probs = deepcopy(s.probs)
    if(length(sp.probs) != 0 && sum(sp.probs) != 0 && !isnan(sum(sp.probs)))
        for hypi in eachindex(m.hyps)
            if(sp.probs[hypi] != 0) 
                for yi in 1:sy
                    for xi in 1:sx
                        if(sp.grid[yi, xi] != os && m.hyps[hypi][yi, xi] != os && m.hyps[hypi][yi, xi] != sp.grid[yi, xi])
                            if(!is_agent(m.hyps[hypi][yi, xi]) && !is_agent(sp.grid[yi, xi]) && !is_reward(m.hyps[hypi][yi, xi]) && !is_reward(sp.grid[yi, xi]) )
                                new_probs[hypi] = 0
                            end
                        end
                    end
                end
            end
        end
    end

    unobserved_count = count_unobserved(sp.grid)
    prev_unobserved_count = count_unobserved(s.grid)
    sp = GridWorldState(sp.grid, sp.ay, sp.ax, sp.d, sp.done, new_probs, sp.reward_count, sp.reward_status, sp.obs_reward_map, sp.obs_reward_key)

    # observed_reward = (prev_unobserved_count-unobserved_count)*m.optimism
    return (sp=sp, o=deepcopy(sp), r=r)
end

# Hack to make julia check if two observations are equal
Base.:(==)(a::GridWorldState, b::GridWorldState) = Base.:(==)(a.grid, b.grid)
function Base.hash(obj::GridWorldState, h::UInt)
    return hash((obj.grid), h)
end


function POMDPs.initialstate(m::Union{GridWorld, GridWorldSSP})
    return ImplicitDistribution() do rng
        s = m.real_observed_state
        unobserved_count = count_unobserved(s.grid)
        reward_status = Vector{GridWorldObs}[]
        for hyp in m.hyps
            push!(reward_status, get_reward_loc(hyp, s.grid))
        end
        return GridWorldState(deepcopy(s.grid), s.ay, s.ax, s.d, false, deepcopy(s.probs), s.reward_count, reward_status, s.obs_reward_map, s.obs_reward_key) 
    end
end

# POMDPs.actions(m::Union{GridWorld, GridWorldSSP}) = [:forward, :forward5, :rotate_left, :rotate_right]
# POMDPs.actions(m::Union{GridWorld, GridWorldSSP}) = [:forward, :backward, :left, :right, :rotate_left, :rotate_right]
POMDPs.actions(m::Union{GridWorld, GridWorldSSP}) = [:forward, :forward5, :backward, :backward5, :left, :left5, :right, :right5]

POMDPs.discount(m::Union{GridWorld, GridWorldSSP}) = m.discount_factor
POMDPs.isterminal(problem::Union{GridWorld, GridWorldSSP}, state::GridWorldState) = state.done


#################################### Julia functional API ####################################

function init_observed_state(state_grid, observation_mode, room_keys, room_key_masks)
    sy, sx = size(state_grid)
    room_keys_matrix = convert(Matrix{Int64}, room_keys)
    state_grid_matrix = convert(Matrix{Int64}, state_grid)
    room_key_masks_matrix = convert(Vector{BitMatrix}, room_key_masks)
    ay, ax, d = get_agent_loc(state_grid_matrix)
    state = GridWorldState(state_grid_matrix, ay, ax, d, false, Float64[], 0, Vector{GridWorldObs}[])
    observed_state = update_observation_from_state(get_unobserved(sy, sx, ay, ax, d), state.grid, Symbol(observation_mode), room_keys_matrix, room_key_masks_matrix)
    return observed_state.grid
end

function next_state(state_grid, observed_state, action, observation_mode, optimism, room_keys, room_key_masks)
    sy, sx = size(state_grid)
    state_grid_matrix = convert(Matrix{Int64}, state_grid)
    room_keys_matrix = convert(Matrix{Int64}, room_keys)
    room_key_masks_matrix = convert(Vector{BitMatrix}, room_key_masks)
    ay, ax, d = get_agent_loc(state_grid_matrix)
    # Take a step in the actual simulation and update the observed state
    current_state = GridWorldState(state_grid_matrix, ay, ax, d, false, Float64[],  0, Vector{GridWorldObs}[])
    current_observed_state = GridWorldState(convert(Matrix{Int64}, observed_state), ay, ax, d, false, Float64[],  0, Vector{GridWorldObs}[])
    next_state, next_observed_state, r = grid_tf(current_state, current_observed_state, Symbol(action), sy, sx, optimism)
    # Update with this observation
    observed_state = update_observation_from_state(next_observed_state, next_state.grid, Symbol(observation_mode), room_keys_matrix, room_key_masks_matrix)
    return next_state.grid, observed_state.grid, r
end



function get_tree_parts(tree)
    # return [tree.total_n, tree.children, tree.o_labels, tree.o_lookup, tree.n, tree.terminalactionnode, tree.v, tree.a_labels]
    filtered_o_labels = []
    for v in tree.o_labels[2:length(tree.o_labels)]
        push!(filtered_o_labels, (v.ay, v.ax) )
    end

    filtered_o_lookup = []
    for ((k, o1), v)  in tree.o_lookup
        push!(filtered_o_lookup, (k, (o1.ay, o1.ax), v))
    end

    return [tree.total_n, tree.children, filtered_o_labels, filtered_o_lookup, tree.n, tree.terminalactionnode, tree.v, tree.a_labels]

end

function step_pomcp(observed_state,
                    max_depth,
                    observation_mode,
                    tree_queries,
                    discount_factor,
                    optimism,
                    obs_reward_map,
                    obs_reward_keys,
                    room_keys,
                    room_key_masks)

    sy, sx = size(observed_state)
    ay, ax, d = get_agent_loc(observed_state)
    # Ensure state grid type
    observed_state = convert(Matrix{Int64}, observed_state)
    tree_queries = convert(Int64, tree_queries)
    obs_reward_key = [GridWorldObs(obs_reward_key[1]+1, obs_reward_key[2]+1) for obs_reward_key in obs_reward_keys]
    room_keys_matrix = convert(Matrix{Int64}, room_keys)
    room_key_masks_matrix = convert(Vector{BitMatrix}, room_key_masks)


    # Set up the POMDP problem
    pomdp = GridWorld(sy, sx,
                      Symbol(observation_mode),
                      GridWorldState(observed_state, ay, ax, d, false, Vector{Bool}[], 0, Vector{GridWorldObs}[], obs_reward_map, obs_reward_key),
                      Vector{Matrix{Int64}}[],
                      discount_factor,
                      optimism,
                      room_keys_matrix,
                      room_key_masks_matrix)

    # Particle filter for multi-step planning
    # Before 0.005
    solver = POMCPSolver(max_depth=max_depth, tree_queries=tree_queries, c=1)

    actions_to_take = []
    planner = solve(solver, pomdp)
    num_unobserved = count_unobserved(observed_state)

    a, info = action_info(planner, initialstate(pomdp), tree_in_info=true)    

    for (s, a, sp, o, r) in stepthrough(pomdp, planner, "s,a,sp,o,r", max_steps=1)
        println("Action: ", a)
        println("Reward: ", r)
        new_num_unobserved = count_unobserved(sp.grid)
        push!(actions_to_take, a)
        if(new_num_unobserved != num_unobserved || r>=1.0)
            break
        end
    end
    tree_parts = get_tree_parts(info[:tree])


    return actions_to_take, tree_parts
end


function step_pomcp_ssp(observed_state, 
                        max_depth, 
                        observation_mode, 
                        tree_queries, 
                        discount_factor, 
                        optimism,
                        hyps, 
                        probs,
                        reward_maps,
                        obs_reward_map,
                        obs_reward_keys,
                        room_keys,
                        room_key_masks)

    sy, sx = size(observed_state)
    ay, ax, d = get_agent_loc(observed_state)
    # Ensure state grid type
    observed_state = convert(Matrix{Int64}, observed_state)
    tree_queries = convert(Int64, tree_queries)
    hyps = convert(Vector{Matrix{Int64}}, hyps)
    probs = convert(Vector{Float64}, probs)
    obs_reward_key = [GridWorldObs(obs_reward_key[1]+1, obs_reward_key[2]+1) for obs_reward_key in obs_reward_keys]
    room_keys_matrix = convert(Matrix{Int64}, room_keys)
    room_key_masks_matrix = convert(Vector{BitMatrix}, room_key_masks)

    if(length(reward_maps)>0)
        if(typeof(reward_maps) == Matrix)
             reward_maps = [reward_maps[i,:] for i in 1:size(reward_maps, 1)]
        end

        reward_maps_full = Vector{Matrix{Float64}}[]
        for reward_map_1 in 1:size(reward_maps, 1)
            if(reward_maps[reward_map_1] != nothing)
                q = Matrix{Float64}[]
                for reward_map_2 in 1:size(reward_maps[reward_map_1], 1)
                    if(reward_maps[reward_map_1][reward_map_2] != nothing)
                    push!(q, reward_maps[reward_map_1][reward_map_2])
                    end
                end
                push!(reward_maps_full, q)
            end 
        end
    else
        reward_maps_full = Vector{Matrix{Float64}}[]
    end

    # Set up the POMDP problem
    pomdp = GridWorldSSP(sy, sx,
                         Symbol(observation_mode),
                         GridWorldState(observed_state, ay, ax, d, false, probs, 0, Vector{GridWorldObs}[], obs_reward_map, obs_reward_key),
                         hyps,
                         reward_maps_full,
                         discount_factor,
                         optimism,
                         room_keys_matrix,
                         room_key_masks_matrix)

    # Particle filter for multi-step planning
    # Before 0.005
    solver = POMCPSolver(max_depth=max_depth, tree_queries=tree_queries, c=1)

    # Set up the POMDP solver
    planner = solve(solver, pomdp)

    # TODO; pass more than the first action and update on new observation or end of traj
    actions_to_take = []
    resulting_states = []
    a, info = action_info(planner, initialstate(pomdp), tree_in_info=true)    
    # inchrome(D3Tree(info[:tree], init_expand=3))

    planner = solve(solver, pomdp)
    num_unobserved = count_unobserved(observed_state)
    for (s, a, sp, o, r) in stepthrough(pomdp, planner, "s,a,sp,o,r", max_steps=1)
        println("Action: ", a)
        println("Reward: ", r)
        new_num_unobserved = count_unobserved(sp.grid)
        push!(actions_to_take, a)
        if(new_num_unobserved != num_unobserved || r>=1.0)
            break
        end
    end
    tree_parts = get_tree_parts(info[:tree])
    return actions_to_take, [], tree_parts
end



####################################### Unused #############################################

function visualize(observed_states::Array)
    fps = 6.0

    function render(t, dt) 
        index = round(Int, t*fps+1)
        
        sy, sz = size(observed_states[index])
        xs = [string("", i) for i = 1:sx]
        ys = [string("", i) for i = 1:sy]
        heatmap(xs, ys, observed_states[index], aspect_ratio = 1, xaxis=false, yaxis=false, colorbar=false) 
    end

    film = roll(render, fps=fps, duration=length(observed_states)/fps)
    write("output.gif", film) # Write to a gif file
end
