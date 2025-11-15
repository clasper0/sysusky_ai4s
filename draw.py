# Drawing a publication-style architecture diagram for the provided GCN+GAT model.
# This code uses matplotlib to draw labeled boxes, arrows and annotations.
# It saves a high-resolution PNG and PDF to /mnt/data so you can download for your paper.
# Follow-up: if you want a different layout, color palette, or SVG export, tell me and I'll adjust.
from matplotlib import pyplot as plt
from matplotlib.patches import FancyBboxPatch, ArrowStyle, FancyArrowPatch, Rectangle, ConnectionPatch
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 600

def draw_box(ax, xy, width, height, text, fontsize=6, boxstyle="round,pad=0.3", color='lightblue'):
    rect = FancyBboxPatch(xy, width, height, boxstyle=boxstyle, linewidth=1.2, 
                         facecolor=color, edgecolor='black', zorder=2)
    ax.add_patch(rect)
    cx = xy[0] + width/2
    cy = xy[1] + height/2
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fontsize, wrap=True)

def draw_arrow(ax, start, end, rad=0.0, lw=0.6, color='black'):
    arrow = FancyArrowPatch(start, end, connectionstyle=f"arc3,rad={rad}", 
                           arrowstyle=ArrowStyle("Simple", head_length=6, head_width=4), 
                           linewidth=lw, zorder=1, color=color)
    ax.add_patch(arrow)

# 增加画布尺寸，提供更多空间
fig = plt.figure(figsize=(16, 10))
ax = fig.add_subplot(111)
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis('off')

# 定义颜色方案
input_color = 'lightyellow'
gcn_color = 'lightblue'
gat_color = 'lightgreen'
pool_color = 'lightcoral'
concat_color = 'plum'
head_color = 'wheat'

# 重新设计布局，增加间距
x0 = 1.0
w_in = 2.0; h_in = 1.0
vertical_spacing = 2.5  # 增加垂直间距

# 输入层 - 移到更高位置
draw_box(ax, (x0, 8.0), w_in, h_in, "Input SMILES\n-> Graph\n(atom features,\nedge_index, edge_attr)", 
         fontsize=3, color=input_color)

# Atom feature block - 增加间距
draw_box(ax, (x0, 5.0), w_in, h_in, "Atom features\n(x, edge_attr)", 
         fontsize=3, color=input_color)

# GCN + GAT stack - 重新布局，增加水平间距
stack_x = 4.0
layer_w = 2.0; layer_h = 1.0; layer_gap = 1.0  # 增加层间距

# 三层GCN+GAT，垂直排列
layer_positions = [8.0, 5.0, 2.0]  # 重新调整Y坐标

for i, y in enumerate(layer_positions):
    # GCN层
    draw_box(ax, (stack_x, y), layer_w, layer_h, f"GCNConv\nHidden {i+1}", 
             fontsize=3, color=gcn_color)
    
    # GAT层 - 移到GCN右侧，增加间距
    gat_x = stack_x + layer_w + 0.5
    draw_box(ax, (gat_x, y + 0.15), 1.4, 0.6, "GATConv\n(attention)", 
             fontsize=2, color=gat_color)

# 残差连接 - 调整箭头位置和弧度
residual_x = stack_x + layer_w + 0.5 + 1.4 + 0.3
draw_arrow(ax, (residual_x, 8.0 - 0.2), (residual_x, 5.0 + layer_h), rad=0.0)
draw_arrow(ax, (residual_x, 5.0 - 0.2), (residual_x, 2.0 + layer_h), rad=0.0)

# Batch norm / activation / dropout - 移到下方
bn_x = stack_x
draw_box(ax, (bn_x, 0.5), layer_w + 2.0, 0.8, 
         "BatchNorm -> Activation -> Dropout\n(inside each layer)", fontsize=3, color='lightgray')

# 池化层 - 向右移动，增加间距
pool_x = 9.0
draw_box(ax, (pool_x, 7.0), 2.0, 1.0, "Attention Pooling\n(global)", 
         fontsize=3, color=pool_color)
draw_box(ax, (pool_x, 4.5), 2.0, 1.0, "Mean / Max / Add\nGlobal Pools", 
         fontsize=3, color=pool_color)

# 从最后一层到池化的箭头 - 调整路径
last_layer_out_x = residual_x
last_layer_y = layer_positions[2] + layer_h/2
draw_arrow(ax, (last_layer_out_x, last_layer_y), (pool_x, 7.5), rad=0.1)
draw_arrow(ax, (last_layer_out_x, last_layer_y), (pool_x, 5.0), rad=-0.2)

# 连接层 - 向右移动
concat_x = 11.5
draw_box(ax, (concat_x, 5.8), 2.0, 1.8, "Concatenate:\n[Attention, Mean, Max, Add]", 
         fontsize=3, color=concat_color)

# 池化到连接的箭头
draw_arrow(ax, (pool_x + 2.0, 7.5), (concat_x, 6.7), rad=0.0)
draw_arrow(ax, (pool_x + 2.0, 5.0), (concat_x, 5.8), rad=0.0)

# 预测头 - 继续向右移动
head_x = 14.0
draw_box(ax, (head_x, 6.0), 1.8, 1.2, "FC Layers\n-> Dropout\n-> Activation\n-> Output", 
         fontsize=3, color=head_color)

draw_arrow(ax, (concat_x + 2.0, 6.7), (head_x, 6.6), rad=0.0)

# 输出分支 - 调整布局
draw_box(ax, (head_x + 2.0, 6.5), 1.8, 0.8, "Output\n(single-task)", 
         fontsize=3, color=head_color)
draw_box(ax, (head_x + 2.0, 4.5), 2.0, 1.2, "Multi-task Heads\n(task1, task2, ...)", 
         fontsize=3, color=head_color)

# 输出箭头
draw_arrow(ax, (head_x + 1.8, 6.6), (head_x + 2.0, 6.9), rad=0.0)
draw_arrow(ax, (head_x + 1.8, 5.8), (head_x + 2.0, 5.1), rad=-0.2)

# 图例/注释 - 移到左下角，不与其他元素重叠
ax.text(0.8, 1.8, "Architecture highlights:", fontsize=5, fontweight='bold')
ax.text(0.8, 1.3, "- GCNConv layers + optional residuals\n- GATConv attention added per-layer\n- BatchNorm / Activation / Dropout\n- Attention + mean/max/add pooling concatenated\n- FC prediction head (single or multi-task)", 
        fontsize=3)

# 输入到堆栈的箭头 - 调整位置
draw_arrow(ax, (x0 + w_in, 8.5), (stack_x, 8.5), rad=0.0)
draw_arrow(ax, (x0 + w_in, 5.5), (stack_x, 5.5), rad=0.0)

# 标题 - 调整位置
ax.text(8, 9.5, "Schematic: Molecular GCN + GAT Architecture", 
        ha='center', fontsize=6, fontweight='bold')

# 保存高分辨率文件
png_path = "/mnt/data/gcn_gat_architecture_spacious.png"
pdf_path = "/mnt/data/gcn_gat_architecture_spacious.pdf"
plt.savefig(png_path, bbox_inches='tight', dpi=600)
plt.savefig(pdf_path, bbox_inches='tight')
plt.show()

png_path, pdf_path