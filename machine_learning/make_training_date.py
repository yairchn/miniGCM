import numpy as np
import netCDF4 as nc
import pylab as plt
import argparse
import os
import fnmatch


# command line:
# python machine_learning/make_training_date.py /Output.HeldSuarez.theta_grid_efold10/Fields/
def main():
    parser = argparse.ArgumentParser(prog='miniGCM')
    parser.add_argument("folder")
    args = parser.parse_args()
    folder = args.folder

    directory = os.getcwd() + folder

    I = 0
    for filename in os.listdir(directory):
        if fnmatch.fnmatch(file, 'Pressure'):
            data = nc.Dataset(directory+filename, 'r')
            A = data.variables['Pressure']
            C = A.reshape(-1,np.shape(A)[2])
            if I==0:
                P=C
            else:
                P=np.vstack([C, C])
            I += 1

    for filename in os.listdir(directory):
        if fnmatch.fnmatch(file, 'Specific_humidity'):
            data = nc.Dataset(directory+filename, 'r')
            A = data.variables['Specific_humidity']
            C = A.reshape(-1,np.shape(A)[2])
            if I==0:
                QT=C
            else:
                QT=np.vstack([C, C])
            I += 1

    for filename in os.listdir(directory):
        if fnmatch.fnmatch(file, 'Temperature'):
            data = nc.Dataset(directory+filename, 'r')
            A = data.variables['Temperature']
            C = A.reshape(-1,np.shape(A)[2])
            if I==0:
                T=C
            else:
                T=np.vstack([C, C])
            I += 1

    for filename in os.listdir(directory):
        if fnmatch.fnmatch(file, 'U_'):
            data = nc.Dataset(directory+filename, 'r')
            A = data.variables['U_']
            C = A.reshape(-1,np.shape(A)[2])
            if I==0:
                U=C
            else:
                U=np.vstack([C, C])
            I += 1

    for filename in os.listdir(directory):
        if fnmatch.fnmatch(file, 'V_'):
            data = nc.Dataset(directory+filename, 'r')
            A = data.variables['V_']
            C = A.reshape(-1,np.shape(A)[2])
            if I==0:
                V=C
            else:
                V=np.vstack([C, C])
            I += 1

    lat = np.array(data.groups['coordinates'].variables['latitude'])
    n = int(np.multiply(data.groups['coordinates'].variables['layers'],1.0))

    lat_list = np.array(data.groups['coordinates'].variables['latitude_list'])
    var = np.array(data.groups['zonal_mean'].variables[varname])
    t = np.divide(data.groups['zonal_mean'].variables['t'],3600.0*24.0)

    X, Y = np.meshgrid(t,lat_list)
    fig = plt.figure(varname)
    for i in range(n):
        ax1 = fig.add_subplot(n, 1, i+1)
        im1 = ax1.contourf(X,Y,np.fliplr(np.rot90(np.squeeze(var[:,:,i]), k=3)))
        ax1.set_ylabel('degree latitude')
        if i<n-1:
            xlabels = [item.get_text() for item in ax1.get_xticklabels()]
            xempty_string_labels = [''] * len(xlabels)
            ax1.set_xticklabels(xempty_string_labels)
        else:
            ax1.set_xlabel('time days')
        fig.colorbar(im1)
    plt.show()

if __name__ == '__main__':
    main()
