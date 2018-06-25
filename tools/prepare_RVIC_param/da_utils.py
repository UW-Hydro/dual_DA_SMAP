
import numpy as np

def read_GIS_ascii_header(file):
    ''' This function reads GIS ascii file header
    Input:
        file: ascii file path; the first 6 lines are the header
    '''

    f = open(file, 'r')
    # ncols
    line = f.readline().rstrip("\n")
    if line.split()[0]!='ncols':
        raise ValueError(
            '{} - flow direction file header variable name' \
            'unsupported!'.format(line.split()[0]))
    ncols = int(line.split()[1])
    # nrows
    line = f.readline().rstrip("\n")
    if line.split()[0]!='nrows':
        raise ValueError(
            '{} - flow direction file header variable name' \
            'unsupported!'.format(line.split()[0]))
    nrows = int(line.split()[1])
    # xllcorner
    line = f.readline().rstrip("\n")
    if line.split()[0]!='xllcorner':
        raise ValueError(
            '{} - flow direction file header variable name' \
            'unsupported!'.format(line.split()[0]))
    xllcorner = float(line.split()[1])
    # yllcorner
    line = f.readline().rstrip("\n")
    if line.split()[0]!='yllcorner':
        raise ValueError(
            '{} - flow direction file header variable name' \
            'unsupported!'.format(line.split()[0]))
    yllcorner = float(line.split()[1])
    # cellsize
    line = f.readline().rstrip("\n")
    if line.split()[0]!='cellsize':
        raise ValueError(
            '{} - flow direction file header variable name' \
            'unsupported!'.format(line.split()[0]))
    cellsize = float(line.split()[1])
    # NODATA_value
    line = f.readline().rstrip("\n")
    if line.split()[0]!='NODATA_value':
        raise ValueError(
            '{} - flow direction file header variable name' \
            'unsupported!'.format(line.split()[0]))
    NODATA_value = float(line.split()[1])

    return ncols, nrows, xllcorner, yllcorner, cellsize, NODATA_value


def generate_xmask_for_route(flowdir_file):
    ''' This function generates xmask (i.e., flow distance) data using haversine formula
    Input:
        Flow direction file path, in the format of 1-8 and 9 for outlet
    Return:
        A np.array matrix of flow distance [unit: m], -1 for inactive grid cells

    Require:
        read_GIS_ascii_header
    '''

    import numpy as np

    r_earth = 6371.0072 * 1000  # earth radius [unit: m]

    #=== Read in flow direction file ===#
    # Read header
    ncols, nrows, xllcorner, yllcorner, cellsize, NODATA_value = \
            read_GIS_ascii_header(flowdir_file)
    # Read flow direction
    fdir_all = np.loadtxt(flowdir_file, dtype=int, skiprows=6)

    #=== Loop over each grid cell in column order ===#
    flow_dist_grid = np.ones([nrows, ncols]) * -1
    lat_max = yllcorner + nrows*cellsize - cellsize/2.0  # Northmost grid cell lat
    for j in range(ncols):
    # Grid cell lon
        grid_lon = xllcorner + cellsize/2.0 + j*cellsize
        for i in range(nrows):
            # Grid cell lat
            grid_lat = lat_max - i*cellsize
            # Calculate flow distance, if active cell
            if fdir_all[i][j]!=int(NODATA_value):  # if active cell
                # Get flow direction
                fdir = fdir_all[i][j]
                # Determine lat and lon of 1st order downstream grid cell
                if fdir==1 or fdir==2 or fdir==8:
                    ds1_lat = grid_lat + cellsize
                elif fdir==4 or fdir==5 or fdir==6:
                    ds1_lat = grid_lat - cellsize
                else:
                    ds1_lat = grid_lat
                if fdir==2 or fdir==3 or fdir==4:
                    ds1_lon = grid_lon + cellsize
                elif fdir==6 or fdir==7 or fdir==8:
                    ds1_lon = grid_lon - cellsize
                else:
                    ds1_lon = grid_lon
                # Calculate flow distance to the downstream grid cell
                hslat = (1 - np.cos((grid_lat-ds1_lat)/180.0*np.pi) ) / 2.0
                hslon = (1 - np.cos((grid_lon-ds1_lon)/180.0*np.pi) ) / 2.0
                flow_dist = 2 * r_earth * np.arcsin(np.sqrt(hslat + \
                                                    np.cos(grid_lat/180.0*np.pi) \
                                                    * np.cos(ds1_lat/180.0*np.pi) * hslon))
                # Save flow distance to array
                flow_dist_grid[i,j] = flow_dist

    return flow_dist_grid

