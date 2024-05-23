# The Routing and Spectrum Allocation Problem

## Motivation
Efficiently solving the Routing and Spectrum Allocation (RSA) problem in optical communication networks is crucial for maximizing resource utilization, improving quality of service, and reducing operational costs. By allocating spectrum and routing optimally, these networks can handle increasing traffic demands, support emerging technologies, and ensure reliable and high-speed communication services for users. Effective solutions to the RSA problem enable optical networks to scale efficiently, minimize congestion, and support the continued growth of digital communication infrastructure.

## The Problem
The objective is to maximize the utilization of optical resources across the network. Let $c(e)$ denote the capacity of a link $e$. When we represent the occupied slots at time $t$ on a link $e$ as $o(e)$, the utilization $u_t$ of the link $e$ at time $t$ can be computed by:
$$u_t(e) := \dfrac{o_t(e)}{c(e)}.$$

Therefore, the average utilization $U(e)$ of a link $e$ over $T$ episodes (equivalent to $T$ arrivals of requests) is
$$U(e) := \dfrac{1}{T}\sum_{t = 0}^{T-1}\dfrac{o_t(e)}{c(e)}.$$
The formal objective is to achieve maximum network-wide utilization. We define the network-wide utilization $U$ as the average of the edge total utility:
$$\text{Maximize } U := \dfrac{1}{|E|}\sum_{e\in E}U(e)$$

The continuity and capacity constraints should be imposed following our Assignment 4.


## Methods
### Routing
- After creating a gymnasium environment, we utilize multiple disjoint paths between arbitrary source and destination nodes
- We use Proximal Policy Optimization (PPO) and Deep Q Network (DQN), a few RL algorithms from the rllib implementation
- We will compare the aforementioned routing technique to the naive approach of only using the shortest path as a way to route a request from the source to the destination node

### Spectrum Allocation
- We choose the available spectrum with lowest index for every request.

## Simulation Settings
- ```nsfnet.gml``` is the optical network topology
- Link capacity is set to 10 for all edges
- We generate 100 (=num_requests) requests that
   - (Case I) All requests have the same source and destination pair (```San Diego Supercomputer Center``` to ```Jon Von Neumann Center, Princeton, NJ```)
   - (Case II) The source and destination nodes are selected uniform-randomly among all nodes. (we make sure $s \neq d$)
   - with a uniformly randomly generated holding time
     - We generate a different holding time for each request by
     - ```np.random.randint(min_ht, max_ht)```
     - Use ```min_ht = 10``` and ```max_ht = 20```
     - For simplicity, we terminate the simulation after accommodating (or blocking) the last ($T$-th) request. No need to wait for all residing requests to leave. We only care about the utilization until the handing of the last request.

## Evaluation
Given that a holding time for any request can range from ```10``` to ```19``` and that there are ```100``` rounds, the expected return from an episode should be ```1,450``` if no blocking is experienced. We see in Case II that DQN converges to this value. Our routing algorithm performs similarly to the simple shortest path algorithm because there is not an overload of requests with the same source and destination pair as seen in Case I.
In Case I, however, our algorithm significantly outperforms the shortest path approach. Obviously, when the number of edges and, subsequently, links are constrained in the network, our algorithm would be a smarter choice to use.

## Paper
- View paper in ```paper``` sub-directory
