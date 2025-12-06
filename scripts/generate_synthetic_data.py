"""
Generate synthetic NFL tracking data for testing the pipeline.
Creates data matching the expected format from the NFL Big Data Bowl competition.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import configs.default as config

np.random.seed(42)


def generate_player_trajectory(start_x, start_y, num_frames, is_receiver=False):
    """Generate a realistic player trajectory."""
    # Base movement with some randomness
    if is_receiver:
        # Receivers run routes - more dramatic movement
        vx = np.random.randn(num_frames) * 2 + 3  # Moving forward
        vy = np.random.randn(num_frames) * 3  # Side to side
    else:
        # Other players - more random movement
        vx = np.random.randn(num_frames) * 1.5
        vy = np.random.randn(num_frames) * 1.5

    # Cumulative position
    x = start_x + np.cumsum(vx * 0.1)  # 0.1 sec per frame
    y = start_y + np.cumsum(vy * 0.1)

    # Keep on field
    x = np.clip(x, 0, 120)
    y = np.clip(y, 0, 53.3)

    # Speed and acceleration
    s = np.sqrt(vx**2 + vy**2)
    a = np.abs(np.diff(s, prepend=s[0])) / 0.1

    return x, y, vx, vy, s, a


def generate_play(game_id, play_id, num_frames=50):
    """Generate a single play with 22 players."""
    rows = []

    # Ball position (center of action)
    ball_x = 50 + np.cumsum(np.random.randn(num_frames) * 0.5)
    ball_y = 26.65 + np.cumsum(np.random.randn(num_frames) * 0.3)

    # Generate 11 offense players
    for i in range(11):
        nfl_id = game_id * 1000 + play_id * 100 + i
        is_receiver = (i == 5)  # One receiver

        start_x = 45 + np.random.rand() * 10
        start_y = 10 + np.random.rand() * 33

        x, y, vx, vy, s, a = generate_player_trajectory(start_x, start_y, num_frames, is_receiver)

        for frame_id in range(num_frames):
            dist_ball = np.sqrt((x[frame_id] - ball_x[frame_id])**2 + (y[frame_id] - ball_y[frame_id])**2)
            rows.append({
                'game_id': game_id,
                'play_id': play_id,
                'nfl_id': nfl_id,
                'frame_id': frame_id + 1,
                'x': x[frame_id],
                'y': y[frame_id],
                'vx': vx[frame_id],
                'vy': vy[frame_id],
                's': s[frame_id],
                'a': a[frame_id],
                'dist_ball': dist_ball,
                'dx_ball': x[frame_id] - ball_x[frame_id],
                'dy_ball': y[frame_id] - ball_y[frame_id],
                'player_role_Targeted Receiver': 1.0 if is_receiver else 0.0,
                'player_role_Defensive Coverage': 0.0,
                'player_side_Offense': 1.0,
                'player_side_Defense': 0.0,
                'play_direction': 'right'
            })

    # Generate 11 defense players
    for i in range(11):
        nfl_id = game_id * 1000 + play_id * 100 + 11 + i
        is_coverage = (i == 5)  # One in coverage

        start_x = 50 + np.random.rand() * 10
        start_y = 10 + np.random.rand() * 33

        x, y, vx, vy, s, a = generate_player_trajectory(start_x, start_y, num_frames, False)

        for frame_id in range(num_frames):
            dist_ball = np.sqrt((x[frame_id] - ball_x[frame_id])**2 + (y[frame_id] - ball_y[frame_id])**2)
            rows.append({
                'game_id': game_id,
                'play_id': play_id,
                'nfl_id': nfl_id,
                'frame_id': frame_id + 1,
                'x': x[frame_id],
                'y': y[frame_id],
                'vx': vx[frame_id],
                'vy': vy[frame_id],
                's': s[frame_id],
                'a': a[frame_id],
                'dist_ball': dist_ball,
                'dx_ball': x[frame_id] - ball_x[frame_id],
                'dy_ball': y[frame_id] - ball_y[frame_id],
                'player_role_Targeted Receiver': 0.0,
                'player_role_Defensive Coverage': 1.0 if is_coverage else 0.0,
                'player_side_Offense': 0.0,
                'player_side_Defense': 1.0,
                'play_direction': 'right'
            })

    return rows


def generate_output_trajectories(input_df, future_frames=50):
    """Generate output trajectories (future positions) for each player."""
    rows = []

    for (game_id, play_id, nfl_id), group in input_df.groupby(['game_id', 'play_id', 'nfl_id']):
        # Get last position
        last_row = group.iloc[-1]
        last_x, last_y = last_row['x'], last_row['y']
        last_vx, last_vy = last_row['vx'], last_row['vy']

        # Generate future trajectory
        for t in range(future_frames):
            # Simple physics with noise
            noise_x = np.random.randn() * 0.5
            noise_y = np.random.randn() * 0.5

            future_x = last_x + (last_vx + noise_x) * (t + 1) * 0.1
            future_y = last_y + (last_vy + noise_y) * (t + 1) * 0.1

            # Keep on field
            future_x = np.clip(future_x, 0, 120)
            future_y = np.clip(future_y, 0, 53.3)

            rows.append({
                'game_id': game_id,
                'play_id': play_id,
                'nfl_id': nfl_id,
                'frame_id': t + 1,
                'x': future_x,
                'y': future_y,
                'play_direction': 'right'
            })

    return pd.DataFrame(rows)


def main():
    print("Generating synthetic NFL tracking data...")

    data_dir = project_root / "data"
    train_dir = data_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    # Generate input data (multiple plays)
    all_input_rows = []
    num_games = 3
    plays_per_game = 10

    for game_id in range(1, num_games + 1):
        for play_id in range(1, plays_per_game + 1):
            print(f"  Generating game {game_id}, play {play_id}...")
            play_rows = generate_play(game_id, play_id, num_frames=30)
            all_input_rows.extend(play_rows)

    input_df = pd.DataFrame(all_input_rows)

    # Save input CSV
    input_path = train_dir / "input_synthetic.csv"
    input_df.to_csv(input_path, index=False)
    print(f"Saved input data: {input_path} ({len(input_df)} rows)")

    # Generate output trajectories
    print("Generating output trajectories...")
    output_df = generate_output_trajectories(input_df, future_frames=config.MAX_TRAJECTORY_LENGTH)

    output_path = train_dir / "output_synthetic.csv"
    output_df.to_csv(output_path, index=False)
    print(f"Saved output data: {output_path} ({len(output_df)} rows)")

    # Create test input (subset)
    test_dir = data_dir / "test_sample"
    test_dir.mkdir(parents=True, exist_ok=True)

    # Use last game as test
    test_input = input_df[input_df['game_id'] == num_games].copy()
    test_input.to_csv(test_dir / "test_input.csv", index=False)
    print(f"Saved test input: {test_dir / 'test_input.csv'}")

    print("\nSynthetic data generation complete!")
    print(f"  Games: {num_games}")
    print(f"  Plays per game: {plays_per_game}")
    print(f"  Total plays: {num_games * plays_per_game}")
    print(f"  Players per play: 22")
    print(f"  Input frames: 30")
    print(f"  Output frames: {config.MAX_TRAJECTORY_LENGTH}")


if __name__ == "__main__":
    main()
