"""
Memory Navigation Rewards - Dopamine for Graph Exploration

Reinforcement signals for DEQ memory navigation:

1. ACCESS REWARD: Using memory vs pure encoder
2. DEPTH REWARD: Multi-hop traversal through graph
3. EFFICIENCY REWARD: Low hyperbolic distance traveled
4. SUCCESS REWARD: Retrieved memory helped prediction

This creates "dopamine" for exploration, encouraging the DEQ
to learn to navigate the memory graph effectively.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryNavigationRewards(nn.Module):
    """
    Computes auxiliary reward signals for memory graph navigation.
    
    These are added to the loss to encourage intelligent exploration
    and efficient traversal of the hyperbolic memory manifold.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Reward coefficients (start small, can be annealed)
        self.access_weight = 0.01      # Reward for using memory
        self.depth_weight = 0.02       # Reward for multi-hop reasoning
        self.efficiency_weight = 0.01  # Reward for short paths
        self.success_weight = 0.05     # Reward when memory helps
        
        print("[Memory Dopamine] Navigation rewards enabled:")
        print(f"  - Access:     {self.access_weight} (use memory > encoder only)")
        print(f"  - Depth:      {self.depth_weight} (multi-hop > single-hop)")
        print(f"  - Efficiency: {self.efficiency_weight} (short path > long path)")
        print(f"  - Success:    {self.success_weight} (helpful > random)")
    
    def compute_access_reward(self, memory_attention, encoder_baseline):
        """
        Reward for accessing memory at all.
        
        Args:
            memory_attention: [B, T, M] - attention over memory nodes
            encoder_baseline: [B, T, C] - what encoder alone would predict
        
        Returns:
            reward: scalar - positive when memory is accessed
        """
        # How much attention is paid to memory? (vs ignoring it)
        # Entropy: High = uniform (exploring), Low = focused (exploiting)
        entropy = -torch.sum(memory_attention * torch.log(memory_attention + 1e-10), dim=-1)
        
        # Reward non-zero access (entropy < max means some focus)
        max_entropy = torch.log(torch.tensor(memory_attention.shape[-1], dtype=torch.float32))
        access_score = 1.0 - (entropy / max_entropy)  # 0 = uniform, 1 = focused
        
        return access_score.mean()
    
    def compute_depth_reward(self, edge_types, memory_attention):
        """
        Reward for multi-hop traversal through graph.
        
        Args:
            edge_types: [B, T, M, k_neighbors, 8] - edge type one-hots
            memory_attention: [B, T, M] - which nodes were accessed
        
        Returns:
            reward: scalar - higher for deeper graph traversal
        """
        # Count edge types used (more types = richer traversal)
        # Weighted by memory attention (only count actually-used edges)
        
        B, T, M, k, num_types = edge_types.shape
        
        # Which edges were actually traversed? (attention to source nodes)
        traversed_edges = memory_attention.unsqueeze(-1).unsqueeze(-1) * edge_types  # [B, T, M, k, 8]
        
        # Count unique edge types used
        type_usage = traversed_edges.sum(dim=[0, 1, 2, 3])  # [8] - usage per type
        num_types_used = (type_usage > 0.01).sum().float()  # How many edge types?
        
        # Normalize: 1-2 types = shallow, 5-8 types = deep multi-hop
        depth_score = num_types_used / num_types
        
        return depth_score
    
    def compute_efficiency_reward(self, edge_weights, memory_attention):
        """
        Reward for low-energy traversal (short paths in hyperbolic space).
        
        Args:
            edge_weights: [B, T, M, k_neighbors] - hyperbolic distances
            memory_attention: [B, T, M] - which nodes accessed
        
        Returns:
            reward: scalar - higher for efficient paths
        """
        # Average hyperbolic distance traveled (weighted by attention)
        # Lower distance = more efficient = higher reward
        
        weighted_distances = (memory_attention.unsqueeze(-1) * edge_weights).sum(dim=2)  # [B, T, k]
        avg_distance = weighted_distances.mean()
        
        # Convert to reward: distance 0 = perfect (reward 1), distance 10 = far (reward 0)
        efficiency_score = torch.exp(-avg_distance / 2.0)  # Exponential decay
        
        return efficiency_score
    
    def compute_success_reward(self, prediction_error, memory_used):
        """
        Reward when memory actually helped reduce prediction error.
        
        This is the KEY signal: "did navigating memory improve the answer?"
        
        Args:
            prediction_error: [B, T] - cross-entropy loss per token
            memory_used: [B, T] - binary flag (1 = used memory, 0 = didn't)
        
        Returns:
            reward: scalar - higher when memory reduces error
        """
        # Compare error when memory was used vs not used
        error_with_memory = (prediction_error * memory_used).sum() / (memory_used.sum() + 1e-10)
        error_without_memory = (prediction_error * (1 - memory_used)).sum() / ((1 - memory_used).sum() + 1e-10)
        
        # Reward: negative = memory helped, positive = memory hurt
        improvement = error_without_memory - error_with_memory
        
        # Clip to [0, 1] and return
        success_score = torch.clamp(improvement, 0.0, 1.0)
        
        return success_score
    
    def compute_total_reward(self, memory_bundle, prediction_error=None, encoder_baseline=None):
        """
        Aggregate all navigation rewards into single scalar.
        
        Args:
            memory_bundle: dict from reflex module with:
                - embeddings: [B, T, M, C]
                - edge_types: [B, T, M, k, 8]
                - edge_weights: [B, T, M, k]
                - attention: [B, T, M] (optional, computed if not present)
            prediction_error: [B, T] (optional, for success reward)
            encoder_baseline: [B, T, C] (optional, for access reward)
        
        Returns:
            total_reward: scalar to SUBTRACT from loss (negative loss = reward)
            reward_dict: breakdown of individual rewards
        """
        if memory_bundle is None:
            return torch.tensor(0.0), {}
        
        rewards = {}
        
        # 1. Access reward (did we use memory at all?)
        if 'attention' in memory_bundle and encoder_baseline is not None:
            access_r = self.compute_access_reward(memory_bundle['attention'], encoder_baseline)
            rewards['access'] = self.access_weight * access_r
        
        # 2. Depth reward (multi-hop traversal?)
        if 'edge_types' in memory_bundle and 'attention' in memory_bundle:
            depth_r = self.compute_depth_reward(memory_bundle['edge_types'], memory_bundle['attention'])
            rewards['depth'] = self.depth_weight * depth_r
        
        # 3. Efficiency reward (short paths?)
        if 'edge_weights' in memory_bundle and 'attention' in memory_bundle:
            efficiency_r = self.compute_efficiency_reward(memory_bundle['edge_weights'], memory_bundle['attention'])
            rewards['efficiency'] = self.efficiency_weight * efficiency_r
        
        # 4. Success reward (did memory help?)
        if prediction_error is not None and 'attention' in memory_bundle:
            # Determine which tokens used memory (attention entropy < threshold)
            attention_entropy = -torch.sum(
                memory_bundle['attention'] * torch.log(memory_bundle['attention'] + 1e-10), 
                dim=-1
            )
            max_entropy = torch.log(torch.tensor(memory_bundle['attention'].shape[-1], dtype=torch.float32))
            memory_used = (attention_entropy < 0.8 * max_entropy).float()  # Focused attention = used
            
            success_r = self.compute_success_reward(prediction_error, memory_used)
            rewards['success'] = self.success_weight * success_r
        
        # Total reward (subtract from loss, so positive reward = lower loss)
        total_reward = sum(rewards.values())
        
        return total_reward, rewards


class MemoryPathReinforcement(nn.Module):
    """
    Reinforcement learning for memory navigation paths.
    
    When a retrieval helps prediction, reinforce the ENTIRE PATH
    that led to that retrieval (not just the final node).
    
    This is graph-based credit assignment!
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Discount factor for path credit (γ in RL)
        self.gamma = 0.9  # 90% credit to each hop back
        
        # Path buffer (store recent successful paths)
        self.path_buffer_size = 100
        self.successful_paths = []
        
        print("[Path Reinforcement] Graph credit assignment enabled:")
        print(f"  - Discount γ: {self.gamma} (credit propagates back through path)")
        print(f"  - Buffer size: {self.path_buffer_size} paths")
    
    def reinforce_path(self, path_nodes, reward):
        """
        Propagate reward backwards through a successful retrieval path.
        
        Args:
            path_nodes: list of node indices [n_0, n_1, ..., n_final]
            reward: scalar reward for final retrieval
        
        Returns:
            path_rewards: [len(path)] rewards for each hop
        """
        path_length = len(path_nodes)
        path_rewards = []
        
        # Discounted reward: r, γr, γ²r, γ³r, ...
        for i in range(path_length):
            hop_reward = reward * (self.gamma ** (path_length - 1 - i))
            path_rewards.append(hop_reward)
        
        return torch.tensor(path_rewards)
    
    def update_edge_weights(self, memory_graph, path_nodes, path_rewards):
        """
        Strengthen edges along successful paths.
        
        Args:
            memory_graph: HybridMemorySystem instance
            path_nodes: list of node IDs traversed
            path_rewards: tensor of rewards for each hop
        """
        # For each edge in path, increase its weight
        for i in range(len(path_nodes) - 1):
            src_node = path_nodes[i]
            dst_node = path_nodes[i + 1]
            reward = path_rewards[i].item()
            
            # Increase edge weight (Hebbian: "neurons that fire together wire together")
            # This is done in the memory system's update_edge() method
            if hasattr(memory_graph, 'reinforce_edge'):
                memory_graph.reinforce_edge(src_node, dst_node, strength=reward)
        
        # Store successful path for analysis
        if len(self.successful_paths) >= self.path_buffer_size:
            self.successful_paths.pop(0)  # Remove oldest
        self.successful_paths.append({
            'path': path_nodes,
            'rewards': path_rewards,
            'total_reward': path_rewards.sum().item()
        })
    
    def get_exploration_bonus(self, node_visit_counts):
        """
        Exploration bonus: encourage visiting less-explored nodes.
        
        Args:
            node_visit_counts: [num_nodes] visit frequency
        
        Returns:
            bonuses: [num_nodes] exploration bonus (higher for rare nodes)
        """
        # Inverse frequency: rare nodes get higher bonus
        # UCB-style: bonus = c / sqrt(visit_count + 1)
        c = 1.0  # Exploration constant
        bonuses = c / torch.sqrt(node_visit_counts + 1)
        
        return bonuses


if __name__ == '__main__':
    """Test navigation rewards"""
    print("Memory Navigation Rewards - Test")
    print("=" * 60)
    
    class MockConfig:
        pass
    
    config = MockConfig()
    rewards = MemoryNavigationRewards(config)
    
    # Mock data
    B, T, M, k = 2, 10, 50, 5
    
    memory_attention = F.softmax(torch.randn(B, T, M), dim=-1)
    edge_types = F.one_hot(torch.randint(0, 8, (B, T, M, k)), num_classes=8).float()
    edge_weights = torch.rand(B, T, M, k) * 5.0  # Distances
    
    memory_bundle = {
        'attention': memory_attention,
        'edge_types': edge_types,
        'edge_weights': edge_weights,
    }
    
    # Compute rewards
    total_reward, reward_dict = rewards.compute_total_reward(memory_bundle)
    
    print(f"\n✓ Total navigation reward: {total_reward.item():.4f}")
    for name, value in reward_dict.items():
        print(f"  - {name}: {value.item():.4f}")
    
    # Test path reinforcement
    print(f"\n" + "=" * 60)
    print("Path Reinforcement - Test")
    print("=" * 60)
    
    path_rl = MemoryPathReinforcement(config)
    
    # Successful path: [node_10, node_23, node_45] with reward 0.8
    path = [10, 23, 45]
    reward = 0.8
    
    path_rewards = path_rl.reinforce_path(path, reward)
    print(f"\n✓ Path: {path}")
    print(f"✓ Final reward: {reward}")
    print(f"✓ Discounted rewards: {path_rewards.tolist()}")
    print(f"  - Hop 0 (10→23): {path_rewards[0]:.3f} (γ² × {reward})")
    print(f"  - Hop 1 (23→45): {path_rewards[1]:.3f} (γ¹ × {reward})")
    print(f"  - Hop 2 (45=final): {path_rewards[2]:.3f} (γ⁰ × {reward})")
