#!/gs/hs0/tga-science/yoda/pymol/bin/python
# -*- Coding: utf-8 -*-
# get images from pymol
# Usage: pymol_getimage_round input_file output_path [n_angles] [-d/--include_diagonal]

import sys
import os
import pymol
import glob
def get_angle(input_file, data_file="./degree.txt"):
    ### NOT USED NOW
    filename = os.path.splitext(input_file)[0]
    with open(data_file, "r") as f:
        data = f.readlines()
        for line in data:
            l = [x for x in line.split(" ") if len(x)>0]
            if l[0] in filename:
                return (int(l[1]), int(l[2]), int(l[3]))

    print("Warning: no degree data found.")
    return (-1, -1, -1)


def rotate_getimage(filename, x=0, y=0,
                    width=256, height=256, dpi=150, ray=0):
    cmd = pymol.cmd
    # rotate xyz
    cmd.rotate([1, 0, 0], angle=x, selection="(all)")
    cmd.rotate([0, 1, 0], angle=y, selection="(all)")

    # save figure
    cmd.png(filename, width=width, height=height, dpi=dpi, ray=ray)

    # rotate back
    cmd.rotate([0, 1, 0], angle=-y, selection="(all)")
    cmd.rotate([1, 0, 0], angle=-x, selection="(all)")


def pymol_getimages_round(input_file, path=".",
                          zoom_buf=8, include_diagonal=False, n_angles=8,
                          width=256, height=256,
                          protein_models=("cartoon", "line", "surface"),
                          surface_transparency=0.7, hbond=True, hbond_radius=0.08,
                          ligand_files=None, docking="glide"):

    cmd = pymol.cmd
    cmd.load(input_file)
    obj = os.path.splitext(input_file)[0]

    # split protein and ligand

    mol_list = cmd.get_object_list(os.path.basename(obj))
    protein = mol_list[0]
    if docking == "glide":
        ligands = mol_list[1:]
    elif docking == "autodock":
        ligands = []
        for f in ligand_files:
            cmd.load(f)
            o = os.path.splitext(f)[0]
            o = os.path.basename(o)
            ligands.append(cmd.get_object_list(o)[0])

    # show protein model
    cmd.hide("all")
    for model in protein_models:
        cmd.show(model, protein)

    if "surface" in protein_models:
        cmd.set("transparency", surface_transparency)
    cmd.set("dash_radius", hbond_radius)

    pymol.util.cbag(protein)

    cmd.zoom(ligands[0], buffer=zoom_buf)
    for lig in ligands:
        num_existfile = len(glob.glob(path + "/" + lig + "_???.png"))
        print(str(lig)+": "+str(num_existfile))
        if num_existfile >= (n_angles**2):
            print("skip " + str(lig))
            continue

        # generate angles
        if include_diagonal:
            # there might be redundant images
            from itertools import product
            d_angles = product([int(360*i/n_angles)
                                for i in range(n_angles)], repeat=2)
        else:
            d_angles = ([(int(360 * i / n_angles), 0) for i in range(n_angles)] +
                        [(0, int(360 * i / n_angles)) for i in range(1, n_angles)])
        #show ligand model
        print(lig)
        cmd.show_as("sticks", lig)
        pymol.util.cbam(lig)

        if hbond:
            cmd.distance("hbond", protein, lig, mode=2)
            cmd.hide("labels")

        # add zoom
        cmd.zoom(lig, buffer=zoom_buf)

        for i, d_angle in enumerate(d_angles):
            angle_x, angle_y = d_angle
            num_pad = "%03d" % i
            filename = path + "/" + lig + "_" + num_pad + ".png"
            rotate_getimage(filename, x=angle_x, y=angle_y,
                            width=width, height=height)
        # reset lig
        cmd.hide(selection=lig)
        if hbond:
            cmd.delete("hbond")

    return


def main():
    # pymol launch: quiet (-q), without GUI (-c)

    pymol.pymol_argv = ['pymol','-qc', '-D', '3']
    pymol.finish_launching()

    input_file = sys.argv[1]
    output_path = sys.argv[2]

    try:
        n_angles = int(sys.argv[3])
    except ValueError:
        n_angles = 8

    include_diagonal = False
    if "--include_diagonal" in sys.argv or "-d" in sys.argv:
        include_diagonal = True

    if input_file.endswith(".pdbqt"):
        ligands = [x for x in sys.argv[3:] if x.endswith(".pdbqt")]
        docking = "autodock"
    elif input_file.endswith("maegz") or input_file.endswith("mae"):
        ligands = None
        docking = "glide"

    pymol_getimages_round(input_file, n_angles=n_angles, include_diagonal=include_diagonal, hbond=False,
                          zoom_buf=32, path=output_path, protein_models=("cartoon", "line"),
                          hbond_radius=0.20, ligand_files=ligands, docking=docking)


if __name__=="__main__":
    main()
    sys.exit(0)