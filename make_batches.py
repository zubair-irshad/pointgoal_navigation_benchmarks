import gzip
import io
import json
import os
import habitat

def make_batches(scenes, out_dir, num_episodes_per_scene,total_batches):
    for i in range(total_batches):
        ep_id =0
        dset_all = habitat.datasets.make_dataset("PointNav-v1")
        for j in range(len(scenes)):
            source_dataset_path = scenes[j]
            with gzip.open(source_dataset_path, "rt") as f:
                deserialized = json.loads(f.read())
            # if j =0 thn 
            dset = habitat.datasets.make_dataset("PointNav-v1")
            dset.episodes = list(deserialized['episodes'][num_episodes_per_scene*i: num_episodes_per_scene*i +num_episodes_per_scene])
            for k, ep in enumerate(dset.episodes):
                ep['episode_id'] = str(ep_id)
                ep_id +=1
            dset_all.episodes.extend(dset.episodes)
                
        out_file = out_dir+'training_batch_'+str(i)+'.json.gz'
        os.makedirs(osp.dirname(out_file), exist_ok=True)
        with gzip.open(out_file, "wt") as f:
            f.write(dset_all.to_json())

def main():

    # Specify scenes directory with the following structure
    # -- train/val/test
    # ------content
    # ----------env_name.json.gz
    scenes  = glob.glob("/home/mirshad7/habitat-api/data/datasets/pointnav/gibson/v1/train/content/*.json.gz")
    out_dir = f'../data/datasets/pointnav/gibson/v1/all_val/'

    num_episodes_per_scene = 2
    total_batches = 100

    make_batches(scenes, out_dir,num_episodes_per_scene,total_batches)

if __name__ == "__main__":
    main()