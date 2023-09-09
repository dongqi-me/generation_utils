import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
# 设置 seaborn 主题
sns.set_theme(style="white")

def calculate_difference(cat1, cat2):
    diff_dict = {}
    for key in cat1:
        if key in cat2:
            diff_dict[key] = cat1[key] - cat2[key]
    return diff_dict

categories = ['pos_tag', 'dep_label', 'named_entity_type', 'discourse_label']


paper_body_stats = data["average_paper_body_stats"]
news_body_stats = data["average_news_body_stats"]

diff_stats = {}
for category in categories:
    diff_stats[category] = calculate_difference(paper_body_stats[category], news_body_stats[category])

fig, axes = plt.subplots(2, 2, figsize=(15, 15), dpi=1200)
axes = axes.flatten()

for idx, category in enumerate(categories):
    keys = list(diff_stats[category].keys())
    values = list(diff_stats[category].values())

    sns.barplot(x=keys, y=values, ax=axes[idx], palette="Spectral")
    axes[idx].tick_params(axis='x', rotation=90)
    # axes[idx].tick_params(axis='y', rotation=45)
    axes[idx].set_xlabel('')
    axes[idx].set_ylabel('')

axes[0].set_title('Absolute Differences in POS Tagging', fontsize=14)
axes[1].set_title('Absolute Differences in Dependency Label', fontsize=14)
axes[2].set_title('Absolute Differences in Named Entity Type', fontsize=14)
axes[3].set_title('Absolute Differences Discourse Relation Label', fontsize=14)

# 移除留白
plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout(pad=0)

# 保存图片
plt.savefig("diff_stats.pdf", bbox_inches='tight', pad_inches=0, dpi=1200)

# plt.show()
