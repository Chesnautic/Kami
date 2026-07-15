; Kami -- Y2K Chaotic Music Visualizer
; Inno Setup installer script.
;
; Expects, relative to this file's directory (the CI workflow sets this up):
;   ..\dist\Kami\        the PyInstaller onedir build (Kami.exe + deps),
;                         with ffmpeg.exe already copied in alongside it
;
; Build with:
;   "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" packaging\installer.iss
;
; Produces:
;   ..\dist_installer\Kami-Setup.exe

#define MyAppName "Kami"
#define MyAppVersion "1.1.2"
#define MyAppPublisher "Kami"
#define MyAppExeName "Kami.exe"

[Setup]
AppId={{6B3B6C0E-6C9F-4A5C-9C7A-3E7B2E9E7A11}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=..\dist_installer
OutputBaseFilename=Kami-Setup
SetupIconFile=..\kami.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
; Installs for the current user only, in a user-writable location -- no
; admin/UAC prompt needed, which matters a lot for "just give people a
; download link" distribution.
PrivilegesRequired=lowest
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Files]
Source: "..\dist\Kami\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName} now"; Flags: nowait postinstall skipifsilent
