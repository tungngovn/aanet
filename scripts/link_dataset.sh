mkdir -p data/KITTI/kitti2012
mkdir -p data/KITTI/kitti2015
mkdir -p data/apolloscape

ln -s /mnt/data/StereoDataset/dataset/kitti2015 ./data/KITTI/kitti2015
ln -s /mnt/data/StereoDataset/dataset/kitti2012 ./data/KITTI/kitti2012
ln -s /mnt/data/StereoDataset/dataset/SceneFlow ./data
ln -s /mnt/data/StereoDataset/dataset/apolloscape ./data/apolloscape
