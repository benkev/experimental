import wx

def get_dir():
    app = wx.App(None)
    style = wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST
    dialog = wx.DirDialog(None, 'Select Flag File Version', style=style, defaultPath='/data/1130643840/034445/ms_vv_rho_c169-170_f08-14_t034500/1130643840_20151104034445_vv_rho_c169-170_f08-14.ms.flagversion')
    if dialog.ShowModal() == wx.ID_OK:
        path = dialog.GetPath()
    else:
        path = None
    dialog.Destroy()
    return path

print get_dir()



