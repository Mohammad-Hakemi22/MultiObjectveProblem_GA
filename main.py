from MOP import MOP



if __name__ == "__main__":
    bga = MOP((0, 0), 0, chrom_l=[0, 0])
    a, b, c, d = bga.run() # start algorithm
    bga.plot(a, b, c, d) # show result and plot