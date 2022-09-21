from nipype.algorithms.confounds import FramewiseDisplacement, TSNR
import PIL

sub = '1'
sess = '2'
rp_path = 'X:\\CompSci\\ResearchProjects\\EJONeill\\Neuroimaging\\foteini\\NFTII\\sub-0'+sub+'\\Session'+sess+'\\CMRR\\rp_CMRR_sub0'+sub+'_sess0'+sess+'.txt'
plot_path = 'fd_plot_sub'+sub+'_sess' + sess +'.png'
calc_path = 'fd_calc_sub'+sub+'_sess' + sess +'.txt'


fd = FramewiseDisplacement()
fd.inputs.in_file = rp_path
fd.inputs.parameter_source = 'SPM'
fd.series_tr = 2.16
fd.out_figure = plot_path
fd.out_file = calc_path
fd.inputs.save_plot= False
res = fd.run()
print("average fd is ..", res.outputs.fd_average)
print("out files is ", res.outputs.out_file )
#print(fd.out_file) 
#print(res.outputs.out_path)




import matplotlib.pyplot as plt
import seaborn as sns

x = []
#load .txt fd so can create a new plot all with same scaling
for line in open('C:/Users/hlw69/fd_power_2012.txt', 'r'):
    if 'Frame' not in line:
        #print(line)
        line =line.replace('\n', '')
        x.append(float(line))
#print(x)
del x[0]


y = list(range(1,len(x)+1))
sns.set_theme(style="whitegrid")
sns.lineplot(y,x,linewidth = 0.7)
plt.axhline(2, ls='--', linewidth=0.8, color='red')
sns.despine(left=True, bottom=True)


t = plt.text(350, 1.7, 'Subject '+sub+' Session ' +sess, fontsize=13)
t.set_bbox(dict(facecolor='silver', alpha=0.5, edgecolor='aliceblue'))

plt.ylim(0, 2.3)
plt.xlim(0, 1100)
plt.xlabel('volume')
plt.ylabel('framewise displacement (FD) in mm')
plt.show()


