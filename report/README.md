# Bridge-RGS 学术报告

- `main.tex`：中文学术论文风格的完整报告源文件
- `refs.bib`：报告引用
- `Bridge-RGS-report.pdf`：已编译 PDF

使用 XeLaTeX 编译（推荐从仓库根目录执行）：

```bash
/usr/lib/chatgpt/resources/tectonic/tectonic -X compile \
  --keep-logs --outdir report/build report/main.tex
```

若本机已安装 TeX Live，也可以使用 `latexmk -xelatex report/main.tex`。报告中的指标来自仓库 `docs/` 的正式 F 版本结果与独立审计回执；主文明确区分留出评价、全量拟合对照和未通过采用门的负面实验。
