' create_desktop_shortcut.vbs
'
' One-time setup: double-click this file to add a "Kami" shortcut to your
' Windows Desktop, using the flower icon (kami.ico) and launching Kami.pyw
' silently through pythonw.exe (no console window popping up).
'
' If double-clicking the shortcut afterwards does nothing, it usually means
' pythonw.exe isn't on your PATH — right-click the new "Kami" shortcut on
' your Desktop, choose Properties, and change the "Target" field to the
' full path of pythonw.exe on your machine, e.g.:
'   C:\Users\<you>\AppData\Local\Programs\Python\Python312\pythonw.exe
' (Target would then be: "<that path>" "<this folder>\Kami.pyw")

Set fso = CreateObject("Scripting.FileSystemObject")
Set WshShell = CreateObject("WScript.Shell")

strScriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
strDesktop = WshShell.SpecialFolders("Desktop")

Set oShellLink = WshShell.CreateShortcut(strDesktop & "\Kami.lnk")
oShellLink.TargetPath = "pythonw.exe"
oShellLink.Arguments = """" & strScriptDir & "\Kami.pyw" & """"
oShellLink.WorkingDirectory = strScriptDir
oShellLink.IconLocation = strScriptDir & "\kami.ico"
oShellLink.Description = "Kami - Y2K Chaotic Music Visualizer"
oShellLink.Save

MsgBox "Kami shortcut added to your Desktop!" & vbCrLf & vbCrLf & _
       "If it doesn't open when you double-click it, see the comment " & _
       "at the top of create_desktop_shortcut.vbs for a quick fix.", _
       64, "Kami setup complete"
