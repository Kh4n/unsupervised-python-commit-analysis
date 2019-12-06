import whatthepatch
import subprocess
import re
import parser
import argparse
import os
import h5py
import numpy as np
import pickle
import keyword
from collections import defaultdict


class RepoParser():
    def __init__(self, git_dir):
        self.git_dir = "--git-dir=" + git_dir
        self.paths = {"counts": defaultdict(int), "embedding": {}, "global_index": 1}
        self.folders = {"counts": defaultdict(int), "embedding": {}, "global_index": 1}
        self.file_types = {"counts": defaultdict(int), "embedding": {}, "global_index": 1}
        self.names_seen = {}
        self.has_loaded_pickle = False
    
    def load_from_pickle(self, pickle_file):
        with open(pickle_file, "rb") as f:
            all_data = pickle.load(f)
            self.paths = all_data["paths"]
            self.folders = all_data["folders"]
            self.file_types = all_data["file_types"]
            self.names_seen = all_data["names_seen"]
        self.has_loaded_pickle = True

    def get_directory_distance(self, path1, path2):
        dirs1 = [k for k in path1.split('/') if k != '']
        dirs2 = [k for k in path2.split('/') if k != '']
        i = 0
        for dir1,dir2 in zip(dirs1, dirs2):
            if dir1 != dir2:
                break
            i += 1
        # print(len(dirs1), len(dirs2), i, dirs2)
        return len(dirs1) - 1 - i + len(dirs2) - 1 - i

    # print(get_directory_distance("a.txt", "/foo/bang/b.txt"))

    def get_average_dir_dist_paths(self, paths):
        if len(paths) == 1:
            return [0]
        ret = []
        for pathA in paths:
            dist_sum = 0
            for pathB in paths:
                if pathA == pathB:
                    continue
                dist_sum += self.get_directory_distance(pathA, pathB)
                # print("dists between", pathA, pathB, get_directory_distance(pathA, pathB))
            ret.append(dist_sum/(len(paths) - 1))
        return ret

    # print(get_average_dir_dist_paths(["foo/bar/baz/a.txt", "foo/bar/baz/b.txt", "foo/bam/baz/c.txt", "wub.txt", "/wub/txt/wub.txt"]))

    def parse_commit(self, diffs, commit_dist_from_origin):
        N_top = 5
        old_paths = []
        new_paths = []

        old_paths_embedding = []
        new_paths_embedding = []

        old_folders_embedding = []
        new_folders_embedding = []

        file_types_embedding = []
        is_new = []

        dist_from_origin = []
        new_name_counts = []
        num_diffs = []
        num_hunks_in_diff = []
        diff_lengths = []
        num_times_file_changed = []
        num_times_folder_changed = []
        num_lines_code_edited = []
        top_n_names_counts = []
        for _ in range(N_top):
            top_n_names_counts.append([])

        diff_count = 0
        parsed_diffs = whatthepatch.parse_patch(diffs)
        for diff in parsed_diffs:
            if diff.changes:
                old_path = diff.header.old_path
                new_path = diff.header.new_path
                old_folder = os.path.dirname(diff.header.old_path)
                new_folder = os.path.dirname(diff.header.new_path)

                ext = os.path.splitext(new_path)[1]
                if ext not in self.file_types["embedding"]:
                    self.file_types["embedding"][ext] = self.file_types["global_index"]
                    self.file_types["global_index"] += 1
                self.file_types["counts"][ext] += 1
                
                file_types_embedding.append(self.file_types["embedding"][ext])

                diff_count += 1
                dist_from_origin.append(commit_dist_from_origin)


                if old_path not in self.paths["embedding"]:
                    self.paths["embedding"][old_path] = self.paths["global_index"]
                    self.paths["global_index"] += 1
                if new_path not in self.paths["embedding"]:
                    self.paths["embedding"][new_path] = self.paths["global_index"]
                    self.paths["global_index"] += 1

                old_paths_embedding.append(self.paths["embedding"][old_path])
                new_paths_embedding.append(self.paths["embedding"][new_path])

                if old_path != new_path:
                    self.paths["counts"][old_path] += 1
                    self.paths["counts"][new_path] += 1
                else:
                    self.paths["counts"][new_path] += 1

                num_times_file_changed.append(self.paths["counts"][new_path])

                old_paths.append(old_path)
                new_paths.append(new_path)



                if old_folder not in self.folders["embedding"]:
                    self.folders["embedding"][old_folder] = self.folders["global_index"]
                    self.folders["global_index"] += 1
                if new_folder not in self.folders["embedding"]:
                    self.folders["embedding"][new_folder] = self.folders["global_index"]
                    self.folders["global_index"] += 1

                old_folders_embedding.append(self.folders["embedding"][old_folder])
                new_folders_embedding.append(self.folders["embedding"][new_folder])

                if old_folder != new_folder:
                    self.folders["counts"][old_folder] += 1
                    self.folders["counts"][new_folder] += 1
                else:
                    self.folders["counts"][new_folder] += 1 

                num_times_folder_changed.append(self.folders["counts"][new_folder])       



                is_new.append(1 if old_path == "/dev/null" else 0)

                diff_lengths.append(len(diff.changes))

                new_name_count = 0
                num_hunks = 0
                lines_code_edited = 0
                name_counts_within_diff = defaultdict(int)

                for change in diff.changes:
                    if change.hunk > num_hunks:
                        num_hunks = change.hunk
                    line = change.line.strip()
                    if len(line) > 0 and line[-1] == ':':
                        line += "\n\tpass"
                    try:
                        a = str(parser.suite(line).tolist())
                        for name in re.findall("\[1, \'(.+?)\'\]", a):
                            if name not in self.names_seen:
                                self.names_seen[name] = 1
                                new_name_count += 1
                            else:
                                self.names_seen[name] += 1
                            if change.old is None and name not in keyword.kwlist:
                                name_counts_within_diff[name] += 1
                        lines_code_edited += 1
                    except Exception as e:
                        # way toooo much printing, so i turned this off after testing it
                        # print("Quietly ignoring exception:", e, line)
                        pass

                top_name_counts = sorted(name_counts_within_diff.items(), key=lambda x: x[1], reverse=True)
                top_name_counts = ([self.names_seen[k] for k,_ in top_name_counts[0:N_top]] + [0]*N_top)[0:N_top]
                for i,k in enumerate(top_name_counts):
                    top_n_names_counts[i].append(k)
                new_name_counts.append(new_name_count)
                num_hunks_in_diff.append(num_hunks)
                num_lines_code_edited.append(lines_code_edited)

        average_dists = self.get_average_dir_dist_paths(new_paths)
        num_diffs = [diff_count]*len(old_paths)

        all_columns = (
            # old_paths,
            # new_paths,

            old_paths_embedding,
            new_paths_embedding,

            old_folders_embedding,
            new_folders_embedding,

            file_types_embedding,
            is_new,

            dist_from_origin,
            new_name_counts,
            num_diffs,
            num_hunks_in_diff,
            diff_lengths,
            num_times_file_changed,
            num_times_folder_changed,
            num_lines_code_edited,
            average_dists,
            *top_n_names_counts,
        )

        lens = [len(k) for k in all_columns]
        assert max(lens) == min(lens)

        return [k for k in zip(*all_columns)]

    def adjust_embedding(self, cur_embedding, cur_stats):
        assert len(cur_embedding) == len(cur_stats)
        stats_sort = sorted(cur_stats.items(), key=lambda x: x[1], reverse=True)
        index_to_new_index = {cur_embedding[k]:(i+1) for i,(k,_) in enumerate(stats_sort)}
        new_embedding = {k:index_to_new_index[cur_embedding[k]] for k,_ in stats_sort}
        return new_embedding, index_to_new_index


    def test(self):
        test_embed =  {'a': 1, 'b': 2, 'c': 3, 'd': 4}
        test_counts = {'a': 6, 'b': 23, 'c': 3, 'd': 10}
        print(self.adjust_embedding(test_embed, test_counts))

    def parse_all_commits(self, out_h5, out_embedding_info, commit_from=None, commit_to="origin"):
        if os.path.isfile(out_embedding_info):
            raise Exception("the output file for the embedding already exists")

        out_h5 = h5py.File(out_h5, 'x')
        out_h5.create_dataset("diff_features", (1000, 20), dtype=np.float32, maxshape=(None, 20))

        if commit_from is None:
            all_commits = subprocess.check_output(["git", self.git_dir, "rev-list", commit_to]).decode("utf-8").splitlines()[:-1]
            commits_from_origin = 0
        else:
            all_commits = subprocess.check_output(["git", self.git_dir, "rev-list", commit_from + ".." + commit_to]).decode("utf-8").splitlines()
            commits_from_origin = int(subprocess.check_output(["git", self.git_dir, "rev-list", "--count", commit_from]))
        all_commits.reverse()
        
        diffs_processed = 0
        for commit in all_commits:
            rows = []
            diffs = subprocess.check_output(["git", self.git_dir, "diff", commit + '~', commit, "-U0"]).decode("ISO-8859-1")
            for row in self.parse_commit(diffs, commits_from_origin):
                if len(row) > 0:
                    out_h5["diff_features"][diffs_processed] = [float(k) for k in row]
                    diffs_processed += 1
                    if diffs_processed % 1000 == 0:
                        out_h5["diff_features"].resize(diffs_processed+1000, 0)
            commits_from_origin += 1
            if commits_from_origin % 100 == 0:
                print("Completed", commits_from_origin, "commits")

        out_h5["diff_features"].resize(diffs_processed, 0)
        print("Completed", commits_from_origin, "commits")

        if self.has_loaded_pickle:
            print("Skipping embedding adjustment to prevent existing embeddings from shifting")
        else:
            print("Retroactively adjusting embeddings based on frequencies...")
            self.paths["embedding"], paths_lut = self.adjust_embedding(self.paths["embedding"], self.paths["counts"])
            self.folders["embedding"], folders_lut = self.adjust_embedding(self.folders["embedding"], self.folders["counts"])
            self.file_types["embedding"], file_types_lut = self.adjust_embedding(self.file_types["embedding"], self.file_types["counts"])
            for i in range(len(out_h5["diff_features"])):
                out_h5["diff_features"][i, 0] = paths_lut[out_h5["diff_features"][i, 0]]
                out_h5["diff_features"][i, 1] = paths_lut[out_h5["diff_features"][i, 1]]

                out_h5["diff_features"][i, 2] = folders_lut[out_h5["diff_features"][i, 2]]
                out_h5["diff_features"][i, 3] = folders_lut[out_h5["diff_features"][i, 3]]

                out_h5["diff_features"][i, 4] = file_types_lut[out_h5["diff_features"][i, 4]]
                if i % 1000 == 0:
                    print("Adjusted", i, "rows")

        out_h5.close()

        with open(out_embedding_info, 'xb') as f:
            all_embedding_data = {"paths": self.paths, "folders": self.folders, "file_types": self.file_types, "names_seen": self.names_seen}
            pickle.dump(all_embedding_data, f)

# test()
def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("git_repo", type=str, help="path to the .git")
    arg_parser.add_argument("out_h5", type=str, help="the output h5 file name")
    arg_parser.add_argument("out_embedding_info", type=str, help="the output file for all the embedding")
    arg_parser.add_argument("--from_commit", type=str, default=None, help="the commit to start from, defaults to initial commit")
    arg_parser.add_argument("--to_commit", type=str, default="origin", help="the commit to go up to, defaults to origin")
    arg_parser.add_argument("--pickle_file", type=str, default=None, help="embedding data to load")
    args = arg_parser.parse_args()

    repo_parser = RepoParser(args.git_repo)
    if args.pickle_file:
        repo_parser.load_from_pickle(args.pickle_file)
    repo_parser.parse_all_commits(args.out_h5, args.out_embedding_info, args.from_commit, args.to_commit)

if __name__ == "__main__":
    main()