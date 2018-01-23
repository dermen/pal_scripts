if has('syntax') && (&t_Co >2)
  syntax on
endif
set hlsearch
set smartindent
set tabstop=4
set shiftwidth=4
set expandtab
set nocp
set number
filetype plugin on

if has("gui_running")
  set hlsearch
  colorscheme blue
  set bs=2
  set ai
  set ruler
endif



