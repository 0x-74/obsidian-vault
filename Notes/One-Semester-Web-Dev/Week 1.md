### LEGEND

| SC  | SELF CLOSING |
|:---:| ------------ |
|     |              |

---

`<br>`  : Line break

`<hr>`  : Horizontal line

`<a>`  : Anchor Element  
- Usage: Hyperlinks, mailing; direct href to files (same directory/relative paths)  
- Attributes:  
  - `target`: Opens in a new tab

`<pre>`  : "What you type is what you see"  
- Retains line breaks, spaces, etc.

`<img>`  : Image Element  
- Attribute:  
  - `alt`: Enhances SEO and accessibility

`<audio ?controls|autoplay|muted|loop>`  : Audio Element  
- Additional info:  
  - `type`: Specifies the audio file type

`<video ?controls|autoplay|muted|loop>`  : Video Element  
- Supported formats:  
  - mp4, webm, ogg

`<link> [SC]`  : Link Element (Self-closing)  
- Attributes:  
  - `rel`: Defines relationship (e.g., "icon")  
  - `type`: Specifies file type and format (e.g., "image/jpg")  
  - `href`: Hyperlink reference

#### Text Formatting Tags:  
- `<b>`, `<i>`, `<u>`, `<del>`, `<big>`, `<small>`  
- `<sub>`, `<super>`, `<tt>`, `<mark>`

### 🧱 Grouping Tags

- `<span>` — Inline grouping element  
- `<div>` — Block-level grouping element  

### 📋 Lists

- `<ul>` — **Unordered List**  
- `<ol>` — **Ordered List**  
- `<dl>` — **Description List**  
  - `<dt>` — **Description Term**  
  - `<dd>` — **Description Definition**  

### 📊 Table Elements

- `<table>` — Defines a table  
	Attributes:
	- `align` — Attribute used to align content (deprecated in HTML5; use CSS instead)  
- `<tr>` — Table row  
- `<th>` — Table header cell  
- `<td>` — Table data cell 


### 🔘 Button Element

- `<button>` — Clickable button  
  - `style` — Inline styling  
  - `onclick` — JavaScript event handler for clicks  
  - Can be **wrapped with an `<a>` (anchor)** tag for navigation behavior  
