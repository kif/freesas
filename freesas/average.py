__author__ = "Guillaume"
__license__ = "MIT"
__copyright__ = "2015, ESRF"

import numpy
from freesas.model import SASModel

class AverModels():
    def __init__(self, filename=None, reference=None):
        self.inputfiles = []
        self.reference = reference if reference is not None else 0#position of reference model in the list of pdb files
        self.outputfile = filename if filename is not None else "aver-model.pdb"
        self.header = []
        self.atoms = []
        self.radius = None
        self.size = []
        self.grid = None

    def __repr__(self):
        return "Average SAS model with %i atoms"%len(self.atoms)

    def models_pooling(self):
        """
        Pool the atoms of each input models in self.atoms
        
        @return self.atoms: coordinates of each atom considerated
        """
        for files in self.inputfiles:
            m = SASModel()
            m.read(files)
            if len(self.atoms)==0:
                self.atoms = m.atoms
            else:
                self.atoms = numpy.append(self.atoms, m.atoms, axis=0)
        return self.atoms

    def calc_radius(self):
        """
        Select the fineness of the reference model and set this value as 2 times self.radius
        
        @return self.radius: radius of an atom in the new grid
        """
        reference = SASModel()
        reference.read(self.inputfiles[self.reference])
        reference._calc_fineness()
        self.radius = 0.5*reference.fineness
        
        return self.radius

    def gridsize(self):
        """
        Calculate the maximal area occupied by models.
        
        @return self.size: 6-list with coordinates x,y,z max and x,y,z min
        """
        if len(self.atoms)==0:
            self.atoms = self.models_pooling()
        atoms = self.atoms
        radius = self.radius
        
        coordmin = atoms.min(axis=0) - 3*radius
        coordmax = atoms.max(axis=0) + 3*radius
        self.size = [coordmax[0],coordmax[1],coordmax[2],coordmin[0],coordmin[1],coordmin[2]]
        
        return self.size

    def makegrid(self):
        """
        Create the new grid points using the area occupied by models.
        
        @return grid: 2d array, coordinates of each point of the grid, fourth column for the occupancy.
        """
        if not self.radius:
            self.calc_radius()
        if not self.size:
            self.size = self.gridsize()
        size = self.size
        radius = self.radius
        
        x = 0.0
        y = 0.0
        z = 0.0
        xlist = []
        ylist = []
        zlist = []
        while x<=size[0]:
            xlist.append(x)
            x += 2*radius
        while y<=size[1]:
            ylist.append(y)
            y += 2*radius
        while z<=size[2]:
            zlist.append(z)
            z += 2*radius
        
        x = - 2*radius
        y = - 2*radius
        z = - 2*radius
        while x>=size[3]:
            xlist.append(x)
            x -= 2*radius
        while y>=size[4]:
            ylist.append(y)
            y -= 2*radius
        while z>=size[5]:
            zlist.append(z)
            z -= 2*radius
        knots = len(xlist)*len(ylist)*len(zlist)
        
        for i in range(len(xlist)):
            for j in range(len(ylist)):
                for k in range(len(zlist)):
                    if i==j==k==0:
                        grid = numpy.array([[xlist[i], ylist[j], zlist[k], 0.0]], dtype="float")
                    else:
                        grid = numpy.append(grid, [[xlist[i], ylist[j], zlist[k], 0.0]], axis=0)
        if grid.shape[0] != knots:
            print "pb with grid lenght"
        self.grid = grid
        return grid

    def trilin_interp(self, atom, gridpoint):
        """
        """
        radius = self.radius
        lattice_len = 2*radius
        lattice_surf = 2*radius*lattice_len
        lattice_vol = 2*radius*lattice_surf
        
        x = atom[0]
        y = atom[1]
        z = atom[2]
        x0 = gridpoint[0]
        y0 = gridpoint[1]
        z0 = gridpoint[2]

        xd = abs(x-x0)
        yd = abs(y-y0)
        zd = abs(z-z0)

        if xd>=2*radius or yd>=2*radius or zd>=2*radius:
            fact = 0.0
        
        elif xd==0 or yd==0 or zd==0:
            if xd==yd==zd==0:
                fact = 1.0
            
            elif xd==yd==0 or yd==zd==0 or xd==zd==0:
                if xd != 0:
                    dist = xd
                elif yd != 0:
                    dist = yd
                else:
                    dist = zd
                fact = dist/(lattice_len)
            
            else:
                if xd == 0:
                    surf = yd * zd
                elif yd == 0:
                    surf = xd * zd
                else:
                    surf = xd * yd
                fact = surf/(lattice_surf)
        else:
            vol = xd*yd*zd
            fact = vol/(lattice_vol)
        return fact


    def assign_occupancy(self):
        """
        Assign an occupancy for each point of the grid.
        
        @return grid: 2d array, fourth column is occupancy of the point
        """
        if len(self.grid)==0:
            self.grid = self.makegrid()
        atoms = self.atoms
        grid = self.grid
        
        for i in range(atoms.shape[0]):
            for j in range(grid.shape[0]):
                fact = self.trilin_interp(atoms[i], grid[j])/len(self.inputfiles)
                grid[j, 3] += fact
        
        order = numpy.argsort(grid, axis=0)[:,-1]
        sortedgrid = numpy.empty_like(grid)
        for i in range(grid.shape[0]):
            sortedgrid[grid.shape[0]-i-1,:] = grid[order[i], :]
        
        self.grid = sortedgrid
        return sortedgrid

    def makeheader(self):
        """
        """
        header = self.header
        
        header.append("number of models averaged : %s"%len(self.inputfiles))
        header.append("filenames :")
        for i in self.inputfiles:
            header.append("--- %s"%i)
        header.append("total number of atoms : %s \n"%self.atoms.shape[0])
        
        for i in range(self.grid.shape[0]):
            x = round(self.grid[i, 0], 3)
            y = round(self.grid[i, 1], 3)
            z = round(self.grid[i, 2], 3)
            occ = round(self.grid[i, 3], 2)
            header.append("ATOM      1  CA  ASP    1       %s  %s  %s  %s 20.0 0 2 201    "%(x, y, z, occ))
            
        for i in range(len(header)):
            print header[i]
        

if __name__ == "__main__":
    aver = AverModels()
    aver.inputfiles = ["aligned-11.pdb"]
    aver.models_pooling()
    aver.calc_radius()
    print "%s atoms"%aver.atoms.shape[0]
    print "radius = %s"%aver.radius
    aver.gridsize()
    print "grid size : ", aver.size
    aver.makegrid()
    print "%s points in the grid"%aver.grid.shape[0]
    aver.assign_occupancy()
    print aver.grid
    print "DONE"