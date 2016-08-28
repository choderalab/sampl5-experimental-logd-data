from openeye.oechem import *
from openeye.oedepict import *

def DrawMoleculeData(cell, mol, df):
    """

    Parameters
    ----------
    cell - cell to draw molecule in
    mol - molecule
    df - dataframe with molecular data
    """
    gap = cell.GetHeight() / 4.0

    font = OEFont()
    font.SetAlignment(OEAlignment_Left)
    font.SetSize(22)
    font.SetFamily(OEFontFamily_Helvetica)
    font.SetStyle(OEFontStyle_Bold)
    data = mol.GetTitle()
    offset = gap
    cell.DrawText(OE2DPoint(0, offset), data, font)

    font.SetStyle(OEFontStyle_Normal)
    offset += gap
    data = "   " + str(df.loc[df['Molecule ID'] == mol.GetTitle()]['log D'].iloc[0])
    cell.DrawText(OE2DPoint(0, offset), data, font)



# Reading csv data because openeye doesnt get SD tags right
import pandas as pd
csvdata = pd.read_csv("sampl5.csv")

ifs = oemolistream()
ifs.SetFlavor(OEFormat_CSV, OEIFlavor_CSV_Header)
ifs.open("sampl5.csv")

image = OEImage(1500, 2000)
image.Clear(OEColor(OEWhite))
mols = iter(ifs.GetOEMols())
rows, cols = 11,10
grid = OEImageGrid(image, rows, cols)
grid.SetMargins(5)
grid.SetCellGap(5)

opts = OE2DMolDisplayOptions(grid.GetCellWidth(), grid.GetCellHeight(), OEScale_AutoScale)
opts.SetTitleLocation(OETitleLocation_Hidden)

col = 0
for r in range(0, grid.NumRows()):
    #depict molecule in first column

    for col in range(5):
        cell = grid.GetCell(r + 1, 1 + 2 * col)

        try:
            mol = next(mols)

            OEPrepareDepiction(mol)
            disp = OE2DMolDisplay(mol, opts)
            OERenderMolecule(cell, disp)
            cell = grid.GetCell(r + 1, 2 + 2 * col)
            DrawMoleculeData(cell, mol, csvdata)
        except StopIteration:
               break

OEWriteImage("SAMPL5data.svg", image)