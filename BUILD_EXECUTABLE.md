# Humor Bot Windows executable build

Copy these four build files into the project root:

- desktop_launcher.py
- HumorBot.spec
- build_humor_exe.ps1
- smoke_test_humor_exe.ps1

Then run:

```powershell
cd D:\Freelancer\Ubuntu\Humour\humor_bot
Set-ExecutionPolicy -Scope Process Bypass
Unblock-File .\build_humor_exe.ps1
.\build_humor_exe.ps1
```

The output is:

```text
dist\HumorBot\HumorBot.exe
```

Use the entire `dist\HumorBot` folder for delivery.

This is intentionally an `onedir` build because PyTorch, Transformers,
tokenizer assets, and the trained model are large. It starts faster and is
more reliable than packing everything into a single self-extracting file.

After building, keep port 8000 free and run:

```powershell
.\smoke_test_humor_exe.ps1
```
