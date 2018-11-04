import sys

# %%-------------------------------------------h.sokooti@gmail.com--------------------------------------------

def IniFun(Ginfo,IN):
    DataPath=Ginfo['DataPath']
    FolderExp=Ginfo['FolderExp']
    RootPath=Ginfo['RootPath']
    ExpN=Ginfo['ExpN']+str(IN)
    RootPathDL=''
    
    if Ginfo['Type']=='Spread':
        NameF=Ginfo['NameF']
        MovingImage=DataPath+NameF[IN-1]+'/followup_1_crop.mha'
        FixedImage=DataPath+NameF[IN-1]+'/baseline_1_crop.mha'
        
    if Ginfo['Type']=='DIR':
        NameF=' '
        MovingImage=DataPath+'/mha/case'+str(IN),'_',Ginfo['TF']+'.mha'
        FixedImage=DataPath+'/mha/case'+str(IN),'_',Ginfo['TM']+'.mha'
        
    if Ginfo['Type']=='SpreadDL':
        RootPathDL=Ginfo['RootPathDL']
        NameF=Ginfo['NameF']
        MovingImage=RootPathDL+ExpN+'/Result/FixedImageRS1.mha'
        FixedImage=RootPathDL,ExpN,'/Result/MovingImageRS1.mha'
        
    if (Ginfo['Swap']>0): 
        FixedImage,MovingImage = MovingImage,FixedImage
        
    return (DataPath,FolderExp,RootPath,ExpN,FixedImage,MovingImage,NameF,RootPathDL)


def MakeGinfo(DLFolder):
    Ginfo = {}
    # Ginfo={'NameF':['NoNameIsGiven']}
    Ginfo['Type'] = 'SpreadDL'
    Ginfo['ExpN'] = 'ExpLung'
    Ginfo['Swap'] = 0
    Ginfo['FolderExp']='LungExp'
    Ginfo['RootPathDL']=DLFolder+'Elastix/'+Ginfo['FolderExp']+'/'
    return Ginfo

class Logger(object):
    # Borrowed from: https://stackoverflow.com/questions/14906764/how-to-redirect-stdout-to-both-file-and-console-with-scripting
    def __init__(self, logAddress=''):
        self.terminal = sys.stdout
        self.log = open(logAddress, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass
