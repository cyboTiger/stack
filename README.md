基于 [Chirpy](https://github.com/cotes2020/jekyll-theme-chirpy) 主题定制化的个人博客，[点击此处进入](https://nihil.cc/)。

[![996.icu](https://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu)

与原版 Chirpy 不同的点：

* 可选用 [Disqus](https://disqus.com/) / [Valine](https://valine.js.org/) / [Waline](https://waline.js.org/) 评论系统，参见 `_config.yml` 里的 `comments` 以及各自的配置段。建议优先考虑使用 Waline 和 Disqus 而不是 Valine。
* 使用[知乎式 404 界面](https://404.life/564.html)，可以返回首页或者返回上一页。
* 新增了分享到 Line，QQ，QQ 空间，贴吧和微博，参见 `_data/share.yml`。
* 使用 [iconfont](https://www.iconfont.cn/) 取代 [Font Awesome](https://fontawesome.com/)，有更多的图标选择空间，参见 `_config.yml` 里的 `iconfont_css`。
* 右侧边栏添加了外部链接块，参见 `_data/external_links.yml`。
* 可以自由地控制在帖子中显示右侧边栏哪些块。参见 `_config.yml` 里的 `panel`。
* 添加了子域页。参见 `_data/subdomain.yml`。不需要该页面的情况下删除 `_tabs/subdomain.md` 即可。
* 添加了 `<details>` 标签的样式，同时修改了 blockquote 的样式。
* 使用改自 [`just the docs`](https://github.com/pmarsceill/just-the-docs) 的表格样式。
* 将代码染色应用到了行内代码段，例如 `` `let fuck_rust = 114514;`{:.language-rust} ``。
* 使用 [Fira Code](https://github.com/tonsky/FiraCode) 作为代码段字体。默认对行内代码不开启连体（Ligature），对块代码开启。
* 暗色主题时，标题和加粗内容在白色文字之中难以区分，增加发光效果以突出显示。
* 运行代码显示输出结果（开发中，目前支持部分语言），需要在代码段的下一行添加 `{: run="lang" }`，例如：

    ````markdown
    ```rust
    fn main() {
        println!("hello world");
    }
    ```
    {: run="rust" }
    ````

    语言支持情况：
    | 已支持的语言 | `run="lang"` 参数 | 后端 |
    | :-: | :-: | :-: |
    | C++ | `run="cpp"` | [Coliru](https://coliru.stacked-crooked.com/) |
    | JavaScript | `run="javascript"` | N/A (本地) |
    | Rust | `run="rust"` | [Rust Playground](https://play.rust-lang.org/) |

如果喜欢我这个定制化的版本，欢迎 Fork，但是请修改 `_config.yml` 中 `google_analytics` 的 `id`， `waline` 的 `server`，以及 `CNAME` 文件中配置的域名，请勿使用我的配置。

通常，每周会至少 merge 一次 [`upstream/master`](https://github.com/cotes2020/jekyll-theme-chirpy) 以追踪新的功能。

原版本的 Chirpy 也会在 [original-chirpy](https://github.com/NichtsHsu/nichtshsu.github.io/tree/original-chirpy) 分支保持同步。
