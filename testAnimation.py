"""import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
plt.style.use('ggplot')

fig, ax = plt.subplots(figsize=(10, 10))
ax.set(xlim=(-3, 3), ylim=(-1, 1))

x = np.linspace(-3, 3, 91)
t = np.linspace(0, 900, 900)
y = np.linspace(-3, 3, 91)
X3, Y3, T3 = np.meshgrid(x, y, t)
sinT3 = np.sin(2*np.pi*T3 )
G = (X3**2 + Y3**2)*sinT3


cax = ax.pcolormesh(x, y, G[:-1, :-1, 0],
                    vmin=-1, vmax=1, cmap='Blues')
fig.colorbar(cax)

def animate(i):
     cax.set_array(G[:-1, :-1, i].flatten())

    
anim = FuncAnimation(
    fig, animate, interval=100, frames=len(t)-1)
 
plt.draw()
plt.show()"""

        
localisations=[]
localisations.append([51.464103, -0.167742,' Clapham Junction'])
localisations.append([51.517081, -0.205802, ' Portobello'])
localisations.append([51.505362, -0.097397,' Bankside'])
localisations.append([51.526960, -0.039702, ' Mile End'])
localisations.append([51.486679, -0.069706,' Avondale'])
localisations.append([51.479221, -0.157701,' Battersea Park'])
localisations.append([51.528112, -0.132139,' Euston'])

localisations.append([51.457496, -0.206173,' East Putney'])
localisations.append([51.530246, -0.183478,' Maida Vale']) ##numéro 10

localisations.append([51.511149, -0.205582,' Ladbroke Grove'])
localisations.append([51.521102, -0.137201," Fitzrovia"])

localisations.append([51.520220, -0.097889,' Barbican'])
localisations.append([51.543253, -0.012612,' Stratford'])
localisations.append([51.452998, -0.169673,' Wandsworth Common'])
localisations.append([51.547134, -0.056992,' Hackney Central'])
localisations.append([51.488620, -0.120795,' Vauxhall'])
localisations.append([51.513266, -0.088799,' Bank'])

localisations.append([51.512528, -0.039737,' Limehouse'])
localisations.append([51.511178, -0.050759,' Shadwell'])
                                            ##effacé:" St.John's Wood"
localisations.append([51.538944, -0.142620,' Camden Town'])
localisations.append([51.497293, -0.081205,' Bermondsey'])
localisations.append([51.503085, -0.203903,' Holland Park'])
localisations.append([51.475048, -0.201434,' Parsons Green'])
localisations.append([51.564610, -0.106011,' Finsbury'])
localisations.append([51.503467, -0.061766,' Wapping'])
localisations.append([51.517235, -0.118733,' Holborn'])
localisations.append([51.530640, -0.028503,' Bow']) 
localisations.append([51.516276, -0.088754,' Moorgate'])
localisations.append([51.538564, -0.075396,' Haggerston'])

localisations.append([51.520168, -0.104948,' Farringdon'])
                                                ##effacé:St Lukes
localisations.append([51.472375, -0.184018,' Sands End'])
localisations.append([51.497346, -0.191145,' Kensington'])

localisations.append([51.487801, -0.167617,' Chelsea']) 

localisations.append([51.495688, -0.219053,' Brook Green']) 
                                            ##effacé: " Parson's Green"
localisations.append([51.526224, -0.108060,' Clerkenwell'])

localisations.append([51.472267, -0.122947,' Stockwell'])#numéro 50: Stockwell station (correct mais déplacé)
localisations.append([51.511759, -0.123452,' Covent Garden']) 
localisations.append([51.497750, -0.020770,' Millwall'])  
localisations.append([51.510528, -0.147792,' Mayfair'])
localisations.append([51.510976, -0.086855,' Monument'])
localisations.append([51.549994, -0.023973,' Hackney Wick'])
localisations.append([51.496434, -0.210585,' Olympia'])
localisations.append([51.504799, -0.218904," Shepherd's Bush"])
localisations.append([51.464036, -0.170242,' Clapham Junction']) ## répétition de clapham plus problme de confusion avec Battersea
localisations.append([51.527094, -0.066881,' Bethnal Green'])
localisations.append([51.491281, -0.224021,' Hammersmith'])
localisations.append([51.461773, -0.138667,' Clapham Common'])
                                            ## supprimé :'Hoxton'
localisations.append([51.488825, -0.205858,' West Kensington'])
localisations.append([51.501025, -0.165279,' Knightsbridge'])

localisations.append([51.538673, -0.082600,' De Beauvoir Town'])

localisations.append([51.531792, -0.125079," King's Cross"])
                                            ## supprimé:" St. Paul's"
localisations.append([51.494497, -0.100889,' Elephant & Castle'])
localisations.append([51.476920, -0.201890,' Fulham']) ##Confondu avec parsons green quasi
localisations.append([51.511232, -0.119261,' Strand'])
localisations.append([51.488872, -0.134937,' Pimlico'])
localisations.append([51.480422, -0.136802,' Mile End']) ##  En double !!
localisations.append([51.532312, -0.157559,"  The Regent's Park"])#75

localisations.append([51.532791, -0.106034,' Angel'])

localisations.append([51.502003, -0.072530,' Shad Thames'])
localisations.append([51.507842, -0.017804,' Poplar'])#numéro 80:' Poplar'
localisations.append([51.522695, -0.163294,' Marylebone'])

                                            ##supprimé: ' Fitzrovia '
localisations.append([51.516377, -0.175567,' Paddington'])
localisations.append([51.475284, -0.124741,' Lambeth'])
localisations.append([51.515118, -0.092141,' Guildhall'])
localisations.append([51.488031, -0.111730,' Kennington'])
localisations.append([51.513354, -0.099828,' St Pauls'])
localisations.append([51.497126, -0.009197,' Cubitt Town'])
localisations.append([51.506860, -0.179122,' Kensington Gardens'])#numéro 90:Kensington Gardens'
localisations.append([51.501810, -0.108770,' Waterloo'])
localisations.append([51.498965, -0.100244,' South Bank'])
localisations.append([51.482008, -0.112953,' Oval'])
localisations.append([51.503456, -0.018831,' Canary Wharf'])
localisations.append([51.510923, -0.114502,' Temple'])
localisations.append([51.516005, -0.047897,' Stepney'])
localisations.append([51.522916, -0.222245])##?? Ladbrooke groove
localisations.append([51.506601, -0.139400," St. James's"])
localisations.append([51.495751, -0.143791,' Victoria'])
localisations.append([51.508172, -0.076314,' Tower'])#numéro 100: 'Tower'
localisations.append([51.495069, -0.183733,' Cromwell Road'])
localisations.append([51.533193, -0.041347,' Old Ford'])#102
localisations.append([51.517540, -0.082986,' Liverpool Street'])

localisations.append([51.493988, -0.173915,' South Kensington'])
localisations.append([51.540551, -0.035536,' Victoria Park'])#107

localisations.append([51.492882, -0.157318,' Sloane Square'])#numéro 110 :' Sloane Square'
localisations.append([51.513731, -0.135650,' Soho'])
localisations.append([51.496818, -0.153294,' Belgravia'])
localisations.append([51.507564, -0.087921,' London Bridge'])
localisations.append([51.517698, -0.210438,' Ladbroke Grove']) ## répétition!!
localisations.append([51.531858, -0.157977," The Regent's Park"])#115 ## répétition !!
localisations.append([51.513472, -0.077418,' Aldgate'])
localisations.append([51.460567, -0.217416,' Putney'])
localisations.append([51.474382, -0.132490,' Wandsworth Road'])
localisations.append([51.491530, -0.193792," Earl's Court"])
localisations.append([51.459037, -0.211727,' East Putney'])#numéro 120 répétition !!
localisations.append([51.508201, -0.095280,' Bankside']) ## répétition
localisations.append([51.512446, -0.233001,' White City'])
localisations.append([51.508066, -0.003763,' Blackwall'])







localisations=[]
localisations.append([51.464103, -0.167742])
localisations.append([51.517081, -0.205802])
localisations.append([51.505362, -0.097397])
localisations.append([51.526960, -0.039702])
localisations.append([51.486679, -0.069706])
localisations.append([51.479221, -0.157701])
localisations.append([51.528112, -0.132139])
localisations.append([51.507306, -0.115475])
localisations.append([51.528709, -0.084792])
localisations.append([51.457496, -0.206173])
localisations.append([51.530246, -0.183478]) ##numéro 10
localisations.append([51.519111, -0.202865])
localisations.append([51.511149, -0.205582])
localisations.append([51.521102, -0.137201])
localisations.append([51.531853, -0.177141])  
localisations.append([51.520220, -0.097889])
localisations.append([51.543253, -0.012612])
localisations.append([51.452998, -0.169673])
localisations.append([51.547134, -0.056992])
localisations.append([51.488620, -0.120795])
localisations.append([51.513266, -0.088799])#numéro 20:' Bank'
localisations.append([51.497493, -0.135626])
                                            #effacé :' Haggerston '
localisations.append([51.483482, -0.196937])
localisations.append([51.512528, -0.039737])
localisations.append([51.511178, -0.050759])
                                            ##effacé:" St.John's Wood"
localisations.append([51.538944, -0.142620])
localisations.append([51.497293, -0.081205])
localisations.append([51.503085, -0.203903])
localisations.append([51.475048, -0.201434])##numéro 30 :correct (parsons green)
localisations.append([51.564610, -0.106011])
localisations.append([51.503467, -0.061766])
localisations.append([51.517235, -0.118733])#33: correct (Holborn)
localisations.append([51.530640, -0.028503]) # !!!! erreur ordonnée: Bow
localisations.append([51.516276, -0.088754])## correct : Moorgate
localisations.append([51.538564, -0.075396])
localisations.append([51.513447, -0.183994])
localisations.append([51.520168, -0.104948])##38 Farringdon : correct
                                                ##effacé:St Lukes
localisations.append([51.472375, -0.184018])#numéro 40: correct (Sands End)
localisations.append([51.497346, -0.191145])# correct :Kensington
localisations.append([51.514891, -0.141562])
localisations.append([51.487801, -0.167617]) # 44 Chelsea ok mais déplacé
localisations.append([51.503055, -0.152623]) 
localisations.append([51.495688, -0.219053]) ## 46 Brook Green correct mais déplacé
                                            ##effacé: " Parson's Green"
localisations.append([51.526224, -0.108060])
localisations.append([51.505233, -0.090851])
localisations.append([51.472267, -0.122947])#numéro 50: Stockwell station (correct mais déplacé)
localisations.append([51.511759, -0.123452]) #Covent garden: correct mais déplacé
localisations.append([51.497750, -0.020770]) ## Milwall 
localisations.append([51.510528, -0.147792])
localisations.append([51.510976, -0.086855])
localisations.append([51.549994, -0.023973])
localisations.append([51.496434, -0.210585])
localisations.append([51.504799, -0.218904])
localisations.append([51.464036, -0.170242])
localisations.append([51.527094, -0.066881])
localisations.append([51.491281, -0.224021])#numéro 60: Hammersmith
localisations.append([51.461773, -0.138667])
                                            ## supprimé :'Hoxton'
localisations.append([51.488825, -0.205858])
localisations.append([51.501025, -0.165279])
localisations.append([51.489929, -0.093154])
localisations.append([51.538673, -0.082600])
localisations.append([51.535645, -0.089741])
localisations.append([51.531792, -0.125079])
                                            ## supprimé:" St. Paul's"
localisations.append([51.494497, -0.100889])#numéro 70:' Elephant & Castle'
localisations.append([51.476920, -0.201890])
localisations.append([51.511232, -0.119261])
localisations.append([51.488872, -0.134937])
localisations.append([51.480422, -0.136802])
localisations.append([51.532312, -0.157559])#75
localisations.append([51.484273, -0.161756])  ##À vérifier (west chelsea)
localisations.append([51.532791, -0.106034])
localisations.append([51.521861, -0.128549])
localisations.append([51.502003, -0.072530])
localisations.append([51.507842, -0.017804])#numéro 80:' Poplar'
localisations.append([51.522695, -0.163294])
localisations.append([51.503687, -0.003279])
                                            ##supprimé: ' Fitzrovia '
localisations.append([51.516377, -0.175567])
localisations.append([51.475284, -0.124741])
localisations.append([51.515118, -0.092141])
localisations.append([51.488031, -0.111730])
localisations.append([51.513354, -0.099828])
localisations.append([51.497126, -0.009197])
localisations.append([51.506860, -0.179122])#numéro 90:Kensington Gardens'
localisations.append([51.501810, -0.108770])
localisations.append([51.498965, -0.100244])
localisations.append([51.482008, -0.112953])
localisations.append([51.503456, -0.018831])
localisations.append([51.510923, -0.114502])
localisations.append([51.516005, -0.047897])
localisations.append([51.522916, -0.222245])
localisations.append([51.506601, -0.139400])
localisations.append([51.495751, -0.143791])
localisations.append([51.508172, -0.076314])#numéro 100: 'Tower'
localisations.append([51.495069, -0.183733])
localisations.append([51.533193, -0.041347])#102
localisations.append([51.517540, -0.082986])
localisations.append([51.531736, -0.134718])
localisations.append([51.515337, -0.065950])
localisations.append([51.493988, -0.173915])
localisations.append([51.534312, -0.041650])#107
                                            ## supprimé= "Shepherd's Bush"
localisations.append([51.476794, -0.189979])
localisations.append([51.492882, -0.157318])#numéro 110 :' Sloane Square'
localisations.append([51.513731, -0.135650])
localisations.append([51.496818, -0.153294])
localisations.append([51.507564, -0.087921])
localisations.append([51.517698, -0.210438])
localisations.append([51.531858, -0.157977])#115
localisations.append([51.513472, -0.077418])
localisations.append([51.460567, -0.217416])
localisations.append([51.474382, -0.132490])
localisations.append([51.491530, -0.193792])
localisations.append([51.459037, -0.211727])#numéro 120
localisations.append([51.508201, -0.095280])
localisations.append([51.512446, -0.233001])