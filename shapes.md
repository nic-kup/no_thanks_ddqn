This markdown file is to help me keep track of the shapes. Since I have a lot of the functions in different places it can be easy to lose track. Hence this file. Let us start with experiences.

# Games (no trajectory)

`singe_game -> List[Tuple[s_t, a_t, r_t, s_{t+1}, done]]`

where `s_t: Array[192]`, `a_t: bool`, `r_t: float`, `done: bool`. But we will later see `a_t: Array[2]` since that is more useful!

```{python3}
sample_from -> Tuple[
    Array[bs, st_size], #s_t
    Array[bs, 2],       #a_t
    Array[bs],          #r_t
    Array[bs, st_size], #s_{t+1}
    Array[bs, 1]        #done
]
```

then the loss function takes in this tuple and does stuff.

# Games (trajectory)

`k_step_game -> List[List[Tuple[...]]]`
Here the inner list is of length `k`, and the `Tuple` is as before.

```
sample_traj -> Tuple[
    Array[bs, traj_len, s],  # s_t, s_t+1, s_{t+k-1}
    Array[bs, traj_len, 2],  # a_t...
    Array[bs, traj_len],     # r_t...
    Array[bs, traj_len, s],  # s_{t+1}, ..., s_{t+k}
    Array[bs, traj_len, 1],  # dones...
]
```